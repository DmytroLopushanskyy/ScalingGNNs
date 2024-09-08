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
            batch_size: int = 1024
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
        if return_edge_id:
            print("return_edge_id time not implemented")

        for edge_type in edge_types:
            src_type, rel_name, dst_type = edge_type if not csc else (edge_type[2], edge_type[1], edge_type[0])
            edge_type = f"{src_type}__{rel_name}__{dst_type}"
            num_hops = len(num_neighbors_dict[edge_type])
            print("num_hops", num_hops)

            # Retrieve seed nodes and neighbours
            num_neighbors = num_neighbors_dict.get(edge_type, [0] * (num_hops + 1))
            print("num_neighbors", num_neighbors)
            seed_nodes = seed_dict.get(src_type)
            if seed_nodes is None:
                continue

            new_nodes = list()
            new_features = list()
            new_labels = list()
            seed_features = list()
            seed_labels = list()
            for batch_start in range(0, len(seed_nodes), batch_size):
                batch_node_ids = seed_nodes[batch_start:batch_start + batch_size]
                # batch_node_ids = [str(x.item()) for x in batch_node_ids]  # Convert to string for the DB

                src_var = src_type.lower()
                query = f"MATCH ({src_var}:{src_type}) \nWHERE {src_var}.id_int IN {batch_node_ids.tolist()} \nWITH {src_var} "
                alias = f"{src_var}"
                neo4j_vars = list()
                hop_nodes_limit = 0
                for hop in range(num_hops):
                    hop_nodes_limit += num_neighbors[hop] * len(batch_node_ids)
                    if num_neighbors[hop] == 0: 
                        continue
                    num = hop + 1
                    for edge_type in edge_types:
                        _, rel_name, dst_type = edge_type if not csc else (edge_type[2], edge_type[1], edge_type[0])
                        relationship_query = f"-[{rel_name.lower()}_{num}:{rel_name}]->({dst_type.lower()}_{num}:{dst_type})"
                        query += f"\nOPTIONAL MATCH ({alias}){relationship_query} "
                        alias = f"{dst_type.lower()}_{num}"
                        neo4j_vars.append(alias)
                        joined_neo4j_vars = ', '.join(neo4j_vars)
                        query += f"\nWITH {src_var}, {joined_neo4j_vars} "
                    query += f"\nORDER BY rand() "
                    query += f"\nWITH {src_var}, {joined_neo4j_vars} LIMIT {hop_nodes_limit} "
                query += f" \nRETURN {src_var}.id_int, " + ", ".join([f"{alias}.id_int, {alias}.label, {alias}.features" for alias in neo4j_vars]) + ";"

                query_response = self._run_query(driver, query)

                src_features, src_labels = self.get_src_attributes(driver, batch_node_ids, src_type)
                print("src attributes retrieved:", src_features.shape, src_labels.shape)
                seed_features.append(src_features)
                seed_labels.append(src_labels)
                    
                for record in query_response:
                    src_id = record.get(f"{src_var}.id_int")
                    for hop_node in neo4j_vars:
                        dst_id = record.get(f"{hop_node}.id_int")
                        if dst_id is None:
                            continue
                        dst_label = record.get(f"{hop_node}.label")
                        dst_features = record.get(f"{hop_node}.features")
                        
                        # Map database IDs to indices
                        src_index = node_id_to_index[src_type].get(src_id)
                        dst_index = node_id_to_index[dst_type].get(dst_id)
                    
                        # Handle new nodes
                        if dst_index is None:
                            if dst_id in new_nodes:
                                dst_index = len(node_id_dict[dst_type]) + new_nodes.index(dst_id) 
                            else:
                                new_nodes.append(int(dst_id))
                                if isinstance(dst_features, str) and ';' in dst_features:
                                    dst_features = [float(x) for x in dst_features.split(';')]
                                new_features.append(dst_features)
                                new_labels.append(int(dst_label) if dst_label is not None else self.empty_label)
                                # Assign the index to the new node position
                                dst_index = len(node_id_dict[dst_type]) + len(new_nodes) - 1

                        # Add edges
                        if src_index is not None:
                            if csc:
                                edge_source_dict[edge_type].append(dst_index)
                                edge_target_dict[edge_type].append(src_index)
                            else:
                                edge_source_dict[edge_type].append(src_index)
                                edge_target_dict[edge_type].append(dst_index)

                            # if return_edge_id:
                            #     edge_id_dict[edge_type].append(rel_id)

            # Consolidate batched data
            seed_features = torch.cat(seed_features, dim=0)
            node_features[src_type] = torch.cat((node_features.get(src_type, torch.tensor([], dtype=torch.float32)), seed_features))
            seed_labels = torch.cat(seed_labels, dim=0)
            node_labels[src_type] = torch.cat((node_labels.get(src_type, torch.tensor([], dtype=torch.float32)), seed_labels))
            edge_source_dict[edge_type] = torch.tensor(edge_source_dict[edge_type], dtype=torch.int64)
            edge_target_dict[edge_type] = torch.tensor(edge_target_dict[edge_type], dtype=torch.int64)
            # if return_edge_id and edge_id_dict[rel_name]:
            #     edge_id_dict[rel_name] = torch.tensor(edge_id_dict[rel_name], dtype=torch.int64)

            # Batch add new nodes to the node dictionary
            print(f"New nodes retrieved for hop {hop}: {len(new_nodes)}")
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

            # Update the number of sampled nodes and edges per hop
            num_sampled_nodes_dict[dst_type].append(len(new_nodes))
            num_sampled_edges_dict[edge_type].append(len(edge_source_dict[edge_type]))

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

    def get_src_attributes(self, driver, node_ids, src_type):
        query = f"MATCH (node:{src_type}) WHERE node.id_int IN $node_ids RETURN node"
        query_response = self._run_query(driver, query, {"node_ids": node_ids})

        src_features_dict = dict()
        src_labels_dict = dict()
        for record in query_response:
            feature_vector = record['node']['features']
            if isinstance(feature_vector, str) and ';' in feature_vector:
                feature_vector = [float(x) for x in feature_vector.split(';')]
            src_features_dict[record['node']["id_int"]] = feature_vector
            src_labels_dict[record['node']["id_int"]] = int(record['node']["label"])

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
