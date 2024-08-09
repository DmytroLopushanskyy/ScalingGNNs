from __future__ import annotations

import multiprocessing
import numpy as np
import torch
from torch_geometric.typing import EdgeTensorType, EdgeType, OptTensor
from torch_geometric.data.graph_store import EdgeAttr, EdgeLayout, GraphStore
from neo4j_remote_backend.client import Neo4jClient


class Neo4jGraphStore(GraphStore):
    def __init__(self, client: Neo4jClient | None = None, num_threads: int | None = None):
        super().__init__()
        if num_threads is None:
            num_threads = multiprocessing.cpu_count()
        if client is None:
            client = Neo4jClient(num_threads)

        self.__client = client
        self.num_threads = num_threads

    def _put_edge_index(self, edge_index: EdgeTensorType, edge_attr: EdgeAttr) -> None:
        print("_put_edge_index", edge_index, edge_attr)

    def _get_edge_index(self, edge_attr: EdgeAttr) -> EdgeTensorType | None:
        """ Returns all edges"""
        print("_get_edge_index", edge_attr)
        # out = self.__client.get_all_edges(edge_attr)
        # print(out.shape)
        # return out
        return None

    def _remove_edge_index(self, edge_attr: EdgeAttr) -> None:
        print("_remove_edge_index", edge_attr)
    
    def get_all_edge_attrs(self) -> list[EdgeAttr]:
        """Returns all registered edge attributes."""
        # print("get_all_edge_attrs", self.__client.get_edge_groups_and_attributes())

        # [EdgeAttr(edge_type=('Paper', 'CITES', 'Paper'), layout=EdgeLayout('coo'), is_sorted=True, size=(2708, 2708))]
        return [EdgeAttr(edge_type=('PAPER', 'CITES', 'PAPER'), layout=EdgeLayout('coo'), is_sorted=True, size=(111059956, 111059956))]
