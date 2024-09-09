from __future__ import annotations

import torch
import numpy as np
from neo4j import GraphDatabase
from torch_geometric.data.feature_store import TensorAttr
from torch_geometric.data.graph_store import EdgeAttr, EdgeLayout, GraphStore
from torch_geometric.typing import FeatureTensorType


class Neo4jClient:
    def __init__(self, num_threads: int = 0):
        uri = "bolt://localhost:7687"
        user = "neo4j"
        password = "password"
        self.__driver = GraphDatabase.driver(uri, auth=(user, password))

    def get_node_groups_and_features(self):
        # return [TensorAttr(group_name='PAPER', attr_name='id', index=None), TensorAttr(group_name='PAPER', attr_name='features', index=None), TensorAttr(group_name='PAPER', attr_name='label', index=None)]
        with self.__driver.session() as session:
            query = """
            MATCH (n)
            WITH DISTINCT labels(n) AS NodeTypes, n
            UNWIND NodeTypes AS NodeType
            WITH NodeType, collect(distinct keys(n)) AS AllKeys
            RETURN NodeType, reduce(s = [], k IN AllKeys | s + k) AS UniqueKeys
            """
            results = session.run(query)
            node_groups_and_features = list()
            for record in results:
                node_type = record["NodeType"]
                for attr in set(record["UniqueKeys"]):
                    node_groups_and_features.append(TensorAttr(node_type, attr))
            print("node_groups_and_features", node_groups_and_features)
            return node_groups_and_features

    def get_edge_groups_and_attributes(self):
        # return [EdgeAttr(edge_type=('Paper', 'CITES', 'Paper'), layout=EdgeLayout('coo'), is_sorted=True, size=(2708, 2708))]
        with self.__driver.session() as session:
            query = """
                        MATCH (a)-[r]->(b)
                        WITH DISTINCT type(r) AS EdgeType, labels(a)[0] AS SourceType, labels(b)[0] AS TargetType, r
                        WITH EdgeType, SourceType, TargetType, collect(distinct keys(r)) AS AllKeys, count(r) AS edge_count
                        RETURN EdgeType, SourceType, TargetType, reduce(s = [], k IN AllKeys | s + k) AS UniqueKeys, edge_count
                        """
            results = session.run(query)
            edge_groups_and_attributes = list()
            for record in results:
                edge_type = (record["SourceType"], record["EdgeType"], record["TargetType"])
                layout = EdgeLayout('coo')
                is_sorted = True
                size = (record["edge_count"], record["edge_count"])
                edge_groups_and_attributes.append(EdgeAttr(
                    edge_type=edge_type,
                    layout=layout,
                    is_sorted=is_sorted,
                    size=size
                ))
            return edge_groups_and_attributes

    def get_tensor_by_query(self, attr: TensorAttr) -> FeatureTensorType | None:
        table_name = attr.group_name
        attr_name = attr.attr_name
        indices = attr.index

        # Convert indices to strings
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        if isinstance(indices, (list, np.ndarray, tuple)):
            indices = [str(int(idx)) for idx in indices]
        elif isinstance(indices, int):
            indices = str(indices)
        elif isinstance(indices, slice):
            indices_start = str(indices.start) if indices.start is not None else None
            indices_stop = str(indices.stop) if indices.stop is not None else None
            indices_step = str(indices.step) if indices.step is not None else None

        match_clause = f"MATCH (item:{table_name})"
        return_clause = f"RETURN item.{attr_name}"

        if indices is None:
            where_clause = ""
        elif isinstance(indices, int):
            where_clause = f"WHERE item.id = {indices}"
        elif isinstance(indices, slice):
            if indices.step is None or indices.step == 1:
                where_clause = f"WHERE item.id >= {indices.start} AND item.id < {indices.stop}"
            else:
                where_clause = (
                    f"WHERE item.id >= {indices.start} AND item.id < {indices.stop} "
                    f"AND (item.id - {indices.start}) % {indices.step} = 0"
                )
        elif isinstance(indices, (torch.Tensor, list, np.ndarray, tuple)):
            where_clause = f"WHERE item.id IN {indices}"
        else:
            msg = f"Invalid attr.index type: {type(indices)!s}"
            raise ValueError(msg)

        query = f"{match_clause} {where_clause} {return_clause}"
        with self.__driver.session() as session:
            result = session.run(query)
            result_list = list()
            for record in result:
                feature_vector = record['item.'+attr_name]
                if isinstance(feature_vector, str) and ';' in feature_vector:
                    feature_vector = [float(x) for x in feature_vector.split(';')]
                result_list.append(feature_vector)

        # Handle stringified ID case and None values
        result_list = [
            int(res) if isinstance(res, str) 
            else (np.nan if res is None else res) 
            for res in result_list
        ]

        return torch.tensor(result_list)

    def get_all_edges(self, edge_attr: EdgeAttr):
        with self.__driver.session() as session:
            # Query to fetch node types, features, and edges
            query = f"""
            MATCH (n)-[r:{edge_attr.edge_type[1]}]->(m)
            WHERE labels(n) IS NOT NULL
            WITH DISTINCT labels(n) AS NodeTypes, n, id(n) AS StartNode, id(m) AS EndNode
            RETURN collect([StartNode, EndNode]) AS Edges
            """
            result = session.run(query)
            edges = []
            for record in result:
                edges.extend(record["Edges"])  # Extend the list with edge pairs

            # Convert edges to a NumPy array and transpose to match the desired shape [2, #of_edges]
            edge_array = np.array(edges).T  # Transpose to get 2 rows and #of_edges columns
            edge_tensor = torch.tensor(edge_array, dtype=torch.long)
            return edge_tensor
