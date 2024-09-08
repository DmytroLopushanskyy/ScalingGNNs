from typing import List, Tuple, Dict, Optional, Union
from collections import defaultdict
import torch
import hashlib
from tqdm import tqdm
from neo4j import GraphDatabase
from torch_geometric.typing import FeatureTensorType
from torch_geometric.data.feature_store import TensorAttr


def hash_tensor(tensor: torch.Tensor) -> str:
    tensor_list = tensor.view(-1).numpy().astype(str).tolist()
    combined_string = '|'.join(tensor_list)
    return hashlib.sha256(combined_string.encode('utf-8')).hexdigest()


class GraphSampler:
    def __init__(self):
        self.node_features = dict()
        self.node_labels = dict()
        self.node_ids = dict()
        self.empty_label = -9223372036854775808

    def _run_query(self, driver, query, parameters=None):
        # Executes a query on the Neo4j database and returns the result.
        with driver.session() as session:
            result = session.run(query, parameters)
            records = list(result)  # Fetch all records before consuming
            return records

    def sample(
            self,
            driver: GraphDatabase.driver,
            node_types: List[str],
            edge_types: List[Tuple[str, str, str]],
            seed_dict: Dict[str, torch.Tensor],  # Seeds for each node type
            num_neighbors_dict: Dict[str, List[int]],
            node_time_dict: Optional[Dict[str, Dict[int, int]]] = None,
            edge_time_dict: Optional[Dict[str, Dict[int, int]]] = None,
            seed_time_dict: Optional[Dict[str, Dict[int, int]]] = None,
            edge_weight_dict: Optional[Dict[str, Dict[int, float]]] = None,
            csc: bool = False,
            replace: bool = False,
            directed: bool = True,
            disjoint: bool = False,
            temporal_strategy: str = '',
            return_edge_id: bool = False,
            batch_size: int = 6000
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], Dict[str, List[int]], Dict[str, List[int]]]:
        print("starting sampling")
        # Initialize dictionaries to store the output data
        edge_source_dict = defaultdict(list)
        edge_target_dict = defaultdict(list)
        edge_id_dict = defaultdict(list) if return_edge_id else None
        node_id_dict = {node_type: seed.clone().detach() for node_type, seed in seed_dict.items()}
        node_features = dict()
        node_labels = dict()
        num_sampled_nodes_dict = defaultdict(list)
        num_sampled_edges_dict = defaultdict(list)

        # Mapping from database node IDs to indices in node_id_dict
        node_id_to_index = {
            node_type: {node_id.item(): idx for idx, node_id in enumerate(seeds)}
            for node_type, seeds in seed_dict.items()
        }

        # Pre-checks for unsupported features
        if disjoint: 
            raise NotImplementedError("Disjoint sampling not implemented")
        if replace: 
            raise NotImplementedError("Sampling with replacement not implemented")
        if edge_weight_dict: 
            raise NotImplementedError("Edge weights not implemented")
        if seed_time_dict: 
            raise NotImplementedError("Seed time not implemented")
        if edge_time_dict: 
            raise NotImplementedError("Edge time not implemented")
        if node_time_dict: 
            raise NotImplementedError("Node time not implemented")

        # Determine the number of hops to perform
        num_hops = max(len(neighbors) for neighbors in num_neighbors_dict.values())
        # Dict to keep track of seed nodes for each hop
        seeds_per_hop = {0: {node_type: seed.clone().detach().tolist() for node_type, seed in seed_dict.items()}}
        
        for hop in range(num_hops):
            for edge_type in edge_types:
                src_type, rel_name, dst_type = edge_type if not csc else (edge_type[2], edge_type[1], edge_type[0])
                relation_name = f"{src_type}__{rel_name}__{dst_type}"

                # Retrieve seed nodes for this hop and source node type
                current_src_nodes = seeds_per_hop[hop].get(src_type, [])
                if not current_src_nodes:
                    continue

                num_neighbors = num_neighbors_dict.get(relation_name, [0] * (hop + 1))[hop]
                print(f"Sampling {num_neighbors} neighbors for edge type {relation_name}, hop {hop} with {len(current_src_nodes)} seeds, batched by {batch_size}")
                if num_neighbors == 0: 
                    continue

                new_nodes = list()
                new_features = list()
                new_labels = list()
                seed_features = list()
                seed_labels = list()
                for batch_start in range(0, len(current_src_nodes), batch_size):
                    batch_node_ids = current_src_nodes[batch_start:batch_start + batch_size]
                    batch_node_ids = [str(x) for x in batch_node_ids]  # Convert to string for the DB
                    results = self._sample_neighbors_from_db(
                        driver,
                        batch_node_ids, src_type, dst_type, num_neighbors,
                        node_time_dict, edge_time_dict, temporal_strategy,
                        replace, directed, disjoint, edge_weight_dict, rel_name, csc, hop
                    )
                    if hop == 0:
                        src_features, src_labels = self.get_src_attributes(driver, batch_node_ids, src_type)
                        print("src attributes retrieved:", src_features.shape, src_labels.shape)
                        seed_features.append(src_features)
                        seed_labels.append(src_labels)
                        
                    for src_id, dst_id, dst_features, dst_labels, rel_id in results:
                        # Map database IDs to indices
                        src_index = node_id_to_index[src_type].get(src_id)
                        dst_index = node_id_to_index[dst_type].get(dst_id)
                    
                        # Handle new nodes
                        if dst_index is None:
                            if dst_id in new_nodes:
                                dst_index = len(node_id_dict[dst_type]) + new_nodes.index(dst_id) 
                            else:
                                new_nodes.append(dst_id)
                                new_features.append(dst_features)
                                new_labels.append(dst_labels)
                                # Temporarily assign the index to the new node position
                                dst_index = len(node_id_dict[dst_type]) + len(new_nodes) - 1

                        # Add edges
                        if src_index is not None:
                            if csc:
                                edge_source_dict[relation_name].append(dst_index)
                                edge_target_dict[relation_name].append(src_index)
                            else:
                                edge_source_dict[relation_name].append(src_index)
                                edge_target_dict[relation_name].append(dst_index)

                            if return_edge_id:
                                edge_id_dict[relation_name].append(rel_id)

                # print(f"New nodes retrieved for hop {hop}: {len(new_nodes)}")

                if seed_features:
                    seed_features = torch.cat(seed_features, dim=0)
                    node_features[src_type] = torch.cat((node_features.get(src_type, torch.tensor([], dtype=torch.float32)), seed_features))
                    # print("#"*20, "src added", len(node_features[src_type]))
                if seed_labels:
                    seed_labels = torch.cat(seed_labels, dim=0)
                    node_labels[src_type] = torch.cat((node_labels.get(src_type, torch.tensor([], dtype=torch.float32)), seed_labels))

                # Batch add new nodes to the node dictionary
                if new_nodes:
                    new_node_tensor = torch.tensor(new_nodes, dtype=torch.int64)
                    new_features_tensor = torch.tensor(new_features, dtype=torch.int64)
                    new_labels_tensor = torch.tensor(new_labels, dtype=torch.int64)
                    
                    start_index = node_id_dict[dst_type].size(0)
                    node_id_dict[dst_type] = torch.cat((node_id_dict[dst_type], new_node_tensor))
                    # print("SHAPE", node_features[dst_type].shape, new_features_tensor.shape)
                    node_features[dst_type] = torch.cat((node_features[dst_type], new_features_tensor))
                    # print("#"*20,"new_features_tensor added", len(new_features_tensor))
                    node_labels[dst_type] = torch.cat((node_labels[dst_type], new_labels_tensor))

                    # Update node_id_to_index with new nodes
                    for idx, node_id in enumerate(new_nodes):
                        node_id_to_index[dst_type][node_id] = start_index + idx

                    seeds_per_hop.setdefault(hop + 1, {}).setdefault(dst_type, []).extend(new_nodes)

                # Update the number of sampled nodes and edges per hop
                num_sampled_nodes_dict[dst_type].append(len(new_nodes))
                num_sampled_edges_dict[relation_name].append(len(edge_source_dict[relation_name]))

        # Convert lists to tensors
        for rel_type in edge_types:
            rel_name = f"{rel_type[0]}__{rel_type[1]}__{rel_type[2]}"
            edge_source_dict[rel_name] = torch.tensor(edge_source_dict[rel_name], dtype=torch.int64)
            edge_target_dict[rel_name] = torch.tensor(edge_target_dict[rel_name], dtype=torch.int64)
            if return_edge_id and edge_id_dict[rel_name]:
                edge_id_dict[rel_name] = torch.tensor(edge_id_dict[rel_name], dtype=torch.int64)

        # Ensure all node types are initialized in node_id_dict
        for node_type in node_types:
            if node_type not in node_id_dict:
                node_id_dict[node_type] = torch.tensor([], dtype=torch.int64)

            # Save all features and labels for future retrieval by the FeatureStore
            hashed_indices = hash_tensor(node_id_dict[node_type])
            if node_type not in self.node_features:
                self.node_features[node_type] = dict()
            if node_type not in self.node_labels:
                self.node_labels[node_type] = dict()
            if node_type not in self.node_ids:
                self.node_ids[node_type] = dict()

            self.node_features[node_type][hashed_indices] = node_features[node_type]
            self.node_labels[node_type][hashed_indices] = node_labels[node_type]
            self.node_ids[node_type][hashed_indices] = node_id_dict[node_type]

        # print("Edge sources:", edge_source_dict)
        # print("Edge targets:", edge_target_dict)
        print("Total Nodes Sampled:", len(node_id_dict['PAPER']))
        # print("node_features", len(self.node_features['PAPER'][hashed_indices]))
        # print("node_labels", len(self.node_labels['PAPER'][hashed_indices]))

        return (
            edge_source_dict, edge_target_dict, node_id_dict,
            edge_id_dict, num_sampled_nodes_dict, num_sampled_edges_dict
        )

    def _sample_neighbors_from_db(
        self, driver, node_ids, src_node_type, dst_node_type, num_samples,
        node_time_dict, edge_time_dict, temporal_strategy, replace, directed,
        disjoint, edge_weight_dict, rel_type, csc, hop
    ):
        # Determine the source and destination based on CSC (Compressed Sparse Column) format
        src_label = dst_node_type if csc else src_node_type
        dst_label = src_node_type if csc else dst_node_type
        src_node = 'dst' if csc else 'src'
        dst_node = 'src' if csc else 'dst'
        
        # Construct the base query for sampling neighbors
        if directed:
            query = f"MATCH ({dst_node}:{dst_label})<-[rel:{rel_type}]-({src_node}:{src_label})"
        else:
            query = f"MATCH ({src_node}:{src_label})-[rel:{rel_type}]-({dst_node}:{dst_label})"

        query += f" WHERE {src_node}.id IN $node_ids "

        # Add temporal strategy conditions if not uniform
        if temporal_strategy != "uniform":
            if node_time_dict and dst_label in node_time_dict:
                query += f" AND {dst_node}.timestamp <= {src_node}.timestamp"
            if edge_time_dict and rel_type in edge_time_dict:
                query += f" AND rel.timestamp <= {src_node}.timestamp"

        query += f"RETURN {src_node}.id, {dst_node}, rel.id LIMIT $num_samples;"

        parameters = {"node_ids": node_ids, "num_samples": num_samples * len(node_ids)}  # this means we can sample more than X per node but on avg no more than X per node
        query_response = self._run_query(driver, query, parameters)
        print("result len", len(query_response))

        results = list()
        for record in query_response:
            dst_feature_vector = record[f"{dst_node}"]["features"]
            if isinstance(dst_feature_vector, str) and ';' in dst_feature_vector:
                dst_feature_vector = [float(x) for x in dst_feature_vector.split(';')]

            results.append((
                int(record[f"{src_node}.id"]),  # src_id
                int(record[f"{dst_node}"]["id"]),  # dst_id
                dst_feature_vector,  # dst_features
                int(record[f"{dst_node}"]["label"]) if record[f"{dst_node}"]["label"] is not None else self.empty_label,
                int(record["rel.id"]) if record["rel.id"] is not None else 0  # rel_id
            ))

        return results

    def get_src_attributes(self, driver, node_ids, src_type):
        query = f"MATCH (node:{src_type}) WHERE node.id IN $node_ids RETURN node"
        query_response = self._run_query(driver, query, {"node_ids": node_ids})

        src_features_dict = dict()
        src_labels_dict = dict()
        for record in query_response:
            feature_vector = record['node']['features']
            if isinstance(feature_vector, str) and ';' in feature_vector:
                feature_vector = [float(x) for x in feature_vector.split(';')]
            src_features_dict[record['node']["id"]] = feature_vector
            src_labels_dict[record['node']["id"]] = int(record['node']["label"])

        src_features_sorted = list()
        src_labels_sorted = list()
        for seed_id in node_ids:
            src_features_sorted.append(src_features_dict.get(seed_id, torch.nan))
            src_labels_sorted.append(src_labels_dict.get(seed_id, self.empty_label))

        return torch.tensor(src_features_sorted, dtype=torch.float32), torch.tensor(src_labels_sorted, dtype=torch.int64)

    def get_tensor(self, attr: TensorAttr) -> Union[FeatureTensorType, None]:
        node_type = attr.group_name
        attr_name = attr.attr_name
        indices = attr.index
        hashed_indices = hash_tensor(indices)

        if attr_name == 'features':
            return self.node_features[node_type][hashed_indices]
        elif attr_name == 'label':
            return self.node_labels[node_type][hashed_indices]
        elif attr_name == 'id':
            # print("self.node_ids", self.node_ids)
            return self.node_ids[node_type][hashed_indices]

        raise ValueError("The attr_name {attr_name} is not supported yet")
