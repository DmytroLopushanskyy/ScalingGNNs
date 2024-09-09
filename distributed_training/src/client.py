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
        # return [TensorAttr(group_name='PRODUCT', attr_name='id', index=None), TensorAttr(group_name='PRODUCT', attr_name='features', index=None), TensorAttr(group_name='PRODUCT', attr_name='label', index=None)]
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
            return node_groups_and_features

    def get_edge_groups_and_attributes(self):
        # return [EdgeAttr(edge_type=('PRODUCT', 'LINK', 'PRODUCT'), layout=EdgeLayout('coo'), is_sorted=True, size=(2449029, 2449029))]
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
                is_sorted = True  # Assume the edges are sorted
                size = (record["edge_count"], record["edge_count"])
                edge_groups_and_attributes.append(EdgeAttr(
                    edge_type=edge_type,
                    layout=layout,
                    is_sorted=is_sorted,
                    size=size
                ))
            return edge_groups_and_attributes

    def get_tensor_by_query(self, attr: TensorAttr) -> FeatureTensorType | None:
        pass

    def get_all_edges(self, edge_attr: EdgeAttr):
        pass
