import torch
from typing import Dict, List, Tuple, Optional, Any, TypedDict
from collections import defaultdict
from neo4j import GraphDatabase
from torch import Tensor


class GraphSampler:
    def _run_query(self, driver, query, parameters=None):
        with driver.session() as session:
            result = session.run(query, parameters)
            records = list(result)  # Fetch all records before consuming
            return records

    def sample(
            self,
            driver: GraphDatabase.driver,
            node_types: List[str],
            edge_types: List[Tuple[str, str, str]],
            seed_dict: Dict[str, List[int]],
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
            batch_size: int = 1
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, list], Optional[dict[Any, Any]], None, None]:
        out_row_dict = {}
        out_col_dict = {}
        out_node_id_dict = {}
        out_edge_id_dict = {} if return_edge_id else None
        num_sampled_nodes_per_hop_dict = None
        num_sampled_edges_per_hop_dict = None

        sampled_nodes_dict = dict()
        mapper_dict = defaultdict(lambda: defaultdict(lambda: -1))
        slice_dict = {}
        seed_times = []

        # Initialize seeds
        for node_type, seeds in seed_dict.items():
            slice_dict[node_type] = (0, len(seeds))
            sampled_nodes_dict[node_type] = seeds
            for i, val in enumerate(seeds):
                mapper_dict[node_type][val] = i

        if disjoint:  # not tested yet
            batch_idx = 0
            for node_type, seeds in seed_dict.items():
                for i, val in enumerate(seeds):
                    sampled_nodes_dict[node_type].append((batch_idx, val))
                    mapper_dict[node_type][(batch_idx, val)] = i
                    batch_idx += 1
                    if seed_time_dict and node_type in seed_time_dict:
                        seed_times.append(seed_time_dict[node_type][val])
                    elif node_time_dict and node_type in node_time_dict:
                        seed_times.append(node_time_dict[node_type][val])

        hops = max(len(v) for v in num_neighbors_dict.values())

        for hop in range(hops):
            for edge_type in edge_types:
                src_node_type = edge_type[0] if not csc else edge_type[2]
                dst_node_type = edge_type[2] if not csc else edge_type[0]
                rel_type = f"{src_node_type}__{edge_type[1]}__{dst_node_type}"
                num_neighbors = num_neighbors_dict[rel_type][hop]
                src_samples = sampled_nodes_dict[src_node_type]
                dst_samples = []

                for i in range(0, len(src_samples), batch_size):
                    node_ids = src_samples[i:i + batch_size].tolist()
                    results = self._sample_neighbors_from_db(
                        driver,
                        node_ids, src_node_type, dst_node_type, num_neighbors,
                        node_time_dict, edge_time_dict, temporal_strategy,
                        replace, directed, disjoint, edge_weight_dict, edge_type[1]
                    )
                    # print(results)
                    dst_samples.extend(results)

                dst_samples_tensor = torch.tensor(dst_samples, dtype=torch.int64)
                sampled_nodes_dict[dst_node_type] = torch.cat((sampled_nodes_dict[dst_node_type], dst_samples_tensor))
                slice_dict[dst_node_type] = (slice_dict[dst_node_type][1], len(sampled_nodes_dict[dst_node_type]))

        for node_type in node_types:
            out_node_id_dict[node_type] = sampled_nodes_dict[node_type]

        if directed:
            for edge_type in edge_types:
                rel_type = f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"
                rows, cols, edges = [], [], []

                for i, src_node in enumerate(
                        sampled_nodes_dict[edge_type[0]] if not csc else sampled_nodes_dict[edge_type[2]]):
                    src_node = int(src_node.item())  # get item from tensor

                    results = self._get_edges_from_db(driver, src_node, edge_type[1], node_time_dict, directed)
                    for neighbor, edge_id in results:
                        if neighbor in mapper_dict[edge_type[2] if not csc else edge_type[0]]:
                            rows.append(i)
                            cols.append(mapper_dict[edge_type[2] if not csc else edge_type[0]][neighbor])
                            if return_edge_id:
                                edges.append(edge_id)

                out_row_dict[rel_type] = torch.tensor(rows, dtype=torch.int64)
                out_col_dict[rel_type] = torch.tensor(cols, dtype=torch.int64)
                if return_edge_id:
                    out_edge_id_dict[rel_type] = torch.tensor(edges, dtype=torch.int64)

        # print("nodes sampled:", len(out_node_id_dict['Paper']))
        return (
            out_row_dict, out_col_dict, out_node_id_dict,
            out_edge_id_dict, num_sampled_nodes_per_hop_dict, num_sampled_edges_per_hop_dict
        )

    def _sample_neighbors_from_db(
            self, driver, node_ids, src_node_type, dst_node_type, num_samples,
            node_time_dict, edge_time_dict, temporal_strategy, replace, directed,
            disjoint, edge_weight_dict, rel_type
    ):
        # Form the base query
        query = f"""
                MATCH (src:{src_node_type})-[rel:{rel_type}]->(dst:{dst_node_type})
                WHERE id(src) IN $node_ids
                """
        parameters = {"node_ids": node_ids, "num_samples": num_samples}

        # Add temporal constraints if necessary
        if node_time_dict and src_node_type in node_time_dict:
            print("node_time_dict and src_node_type in node_time_dict")
            query += " AND dst.timestamp <= $timestamp"
            parameters["timestamp"] = max(node_time_dict[src_node_type].values())

        if edge_time_dict and rel_type in edge_time_dict:
            print("edge_time_dict and rel_type in edge_time_dict")
            query += " AND rel.timestamp <= $edge_timestamp"
            parameters["edge_timestamp"] = max(edge_time_dict[rel_type].values())

        # Randomly sample neighbors with or without replacement
        query += " RETURN id(dst), rand() as r"
        if edge_weight_dict and rel_type in edge_weight_dict:
            print("edge_weight_dict and rel_type in edge_weight_dict")
            query += ", rel.weight as weight"

        query += " ORDER BY r LIMIT $num_samples"

        result = self._run_query(driver, query, parameters)
        # print(result)
        return [record["id(dst)"] for record in result]

    def _get_edges_from_db(self, driver, src_node_id, rel_type, node_time_dict, directed):
        query = f"""
                MATCH (src)-[rel:{rel_type}]->(dst)
                WHERE id(src) = $src_node_id
                """

        parameters = {"src_node_id": src_node_id}

        # Add temporal constraints if necessary
        if node_time_dict:
            query += " AND dst.timestamp <= $timestamp"
            parameters["timestamp"] = max(node_time_dict.values())

        query += " RETURN id(dst), id(rel)"
        result = self._run_query(driver, query, parameters)
        return [(record["id(dst)"], record["id(rel)"]) for record in result]
