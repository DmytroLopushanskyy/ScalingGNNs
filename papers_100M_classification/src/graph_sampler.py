from typing import List, Tuple, Dict, Optional, Union
from collections import defaultdict
import torch
import hashlib
from tqdm import tqdm
from neo4j import GraphDatabase
from torch_geometric.typing import FeatureTensorType
from torch_geometric.data.feature_store import TensorAttr


def hash_indices(indices: List[str]) -> str:
    combined_string = '|'.join(indices)
    hash_object = hashlib.sha256(combined_string.encode('utf-8'))
    return hash_object.hexdigest()


class GraphSampler:
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
            batch_size: int = 1000
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], Dict[str, List[int]], Dict[str, List[int]]]:
        # Initialize dictionaries to store the output data
        edge_source_dict = defaultdict(list)
        edge_target_dict = defaultdict(list)
        edge_id_dict = defaultdict(list) if return_edge_id else None
        node_id_dict = {node_type: seed.clone().detach() for node_type, seed in seed_dict.items()}
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
        seeds_per_hop = {0: {node_type: seed.clone().detach().tolist() for node_type, seed in seed_dict.items()}}

        for hop in range(num_hops):
            for edge_type in edge_types:
                # Define source, destination, and relation type
                src_type, rel_name, dst_type = edge_type if not csc else (edge_type[2], edge_type[1], edge_type[0])
                relation_name = f"{src_type}__{rel_name}__{dst_type}"

                current_src_nodes = seeds_per_hop[hop].get(src_type, [])
                if not current_src_nodes:
                    continue

                num_neighbors = num_neighbors_dict.get(relation_name, [0] * (hop + 1))[hop]
                print(f"Sampling {num_neighbors} neighbors for edge type {relation_name}, hop {hop}")

                if num_neighbors == 0: 
                    continue

                new_nodes = list()
                for batch_start in range(0, len(current_src_nodes), batch_size):
                    batch_node_ids = current_src_nodes[batch_start:batch_start + batch_size]
                    results = self._sample_neighbors_from_db(
                        driver,
                        batch_node_ids, src_type, dst_type, num_neighbors,
                        node_time_dict, edge_time_dict, temporal_strategy,
                        replace, directed, disjoint, edge_weight_dict, rel_name, csc, hop
                    )
                    for src_id, dst_id, rel_id in results:
                        # Map database IDs to indices
                        src_index = node_id_to_index[src_type].get(src_id)
                        dst_index = node_id_to_index[dst_type].get(dst_id)

                        # Handle new nodes
                        if dst_index is None:
                            if dst_id in new_nodes:
                                dst_index = len(node_id_dict[dst_type]) + new_nodes.index(dst_id) 
                            else:
                                new_nodes.append(dst_id)
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

                print(f"New Nodes: {len(new_nodes)}", "hop", hop)

                # Batch add new nodes to the node dictionary
                if new_nodes:
                    new_node_tensor = torch.tensor(new_nodes, dtype=torch.int64)
                    start_index = node_id_dict[dst_type].size(0)
                    node_id_dict[dst_type] = torch.cat((node_id_dict[dst_type], new_node_tensor))

                    # Update node_id_to_index with new nodes
                    for idx, node_id in enumerate(new_nodes):
                        node_id_to_index[dst_type][node_id] = start_index + idx

                    seeds_per_hop.setdefault(hop + 1, {}).setdefault(dst_type, []).extend(new_nodes)

                # Update the number of sampled nodes and edges per hop
                num_sampled_nodes_dict[dst_type].append(len(new_nodes))
                num_sampled_edges_dict[relation_name].append(len(edge_source_dict[relation_name]))

                print(f"end of edge sampling", hop)
            print(f"end of hop {hop}")

        print("here1")

        # Convert lists to tensors
        for rel_type in edge_types:
            rel_name = f"{rel_type[0]}__{rel_type[1]}__{rel_type[2]}"
            edge_source_dict[rel_name] = torch.tensor(edge_source_dict[rel_name], dtype=torch.int64)
            edge_target_dict[rel_name] = torch.tensor(edge_target_dict[rel_name], dtype=torch.int64)
            if return_edge_id and edge_id_dict[rel_name]:
                edge_id_dict[rel_name] = torch.tensor(edge_id_dict[rel_name], dtype=torch.int64)

        print("here2")

        # Ensure all node types are initialized in node_id_dict
        for node_type in node_types:
            if node_type not in node_id_dict:
                node_id_dict[node_type] = torch.tensor([], dtype=torch.int64)

        # print("Edge sources:", edge_source_dict)
        # print("Edge targets:", edge_target_dict)
        print("node_id_dict:", len(node_id_dict['PAPER']))
        print("return")

        return (
            edge_source_dict, edge_target_dict, node_id_dict,
            edge_id_dict, num_sampled_nodes_dict, num_sampled_edges_dict
        )

    def _sample_neighbors_from_db(
        self, driver, node_ids, src_node_type, dst_node_type, num_samples,
        node_time_dict, edge_time_dict, temporal_strategy, replace, directed,
        disjoint, edge_weight_dict, rel_type, csc, hop
    ):
        # Construct the base query for sampling neighbors
        if directed:
            if csc:
                query = f"MATCH (dst:{dst_node_type})<-[rel:{rel_type}]-(src:{src_node_type})"
            else:
                query = f"MATCH (src:{src_node_type})-[rel:{rel_type}]->(dst:{dst_node_type})"
        else:
            query = f"MATCH (src:{src_node_type})-[rel:{rel_type}]-(dst:{dst_node_type})"

        query += f" WHERE {'dst' if csc else 'src'}.id IN $node_ids"

        # Add temporal strategy conditions if not uniform
        if temporal_strategy != "uniform":
            if node_time_dict and dst_node_type in node_time_dict:
                query += " AND dst.timestamp <= src.timestamp"
            if edge_time_dict and rel_type in edge_time_dict:
                query += " AND rel.timestamp <= src.timestamp"

        # query += f"""
        #         WITH {'dst' if csc else 'src'}, collect({'src' if csc else 'dst'}) AS neighbors, collect(rel) AS rels
        #         CALL {{
        #             WITH neighbors, rels
        #             UNWIND range(0, size(neighbors) - 1) AS idx
        #             RETURN neighbors[idx] AS neighbor, rels[idx] AS edge, rand() AS r
        #             ORDER BY r
        #             LIMIT $num_samples
        #         }}
        #         RETURN {'dst.id AS src_id, neighbor.id AS dst_id' if csc else 'src.id AS src_id, neighbor.id AS dst_id'}, edge.id AS rel_id;
        #         """
        query += "RETURN"
        if hop == 0: 
            # If this is the first hop, we need to extract all source attributes
            # In the next hops, only destinations' attributes will be requested
            query += " src.* "
        else:
            query += " src.id "

        query += f"""
                dst.*, rel.id
                LIMIT $num_samples;
                """
        
        node_ids = [str(x) for x in node_ids]
        parameters = {"node_ids": node_ids, "num_samples": num_samples * len(node_ids)}  # this means we can sample more than 12 per node but on avg no more than 12 per node

        result = self._run_query(driver, query, parameters)
        print("result len", len(result))

        return [(int(record["src_id"]), int(record["dst_id"]), int(record["rel_id"]) if record["rel_id"] is not None else 0) for record in result]

    def get_tensor(self, attr: TensorAttr) -> Union[FeatureTensorType, None]:
        table_name = attr.group_name
        attr_name = attr.attr_name
        indices = attr.index

        print("indices", len(indices))
        raise ValueError()

sampler = GraphSampler()