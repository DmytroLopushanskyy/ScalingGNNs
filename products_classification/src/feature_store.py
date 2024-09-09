from __future__ import annotations

import multiprocessing

import numpy as np
import torch
from torch_geometric.typing import FeatureTensorType
from torch_geometric.data.feature_store import TensorAttr, FeatureStore
from src.client import Neo4jClient

from src.graph_sampler import GraphSampler


class Neo4jFeatureStore(FeatureStore):
    def __init__(self, client: Neo4jClient | None = None, sampler: GraphSampler = None, num_threads: int | None = None):
        super().__init__()
        if num_threads is None:
            num_threads = multiprocessing.cpu_count()
        if client is None:
            client = Neo4jClient(num_threads)

        self.client = client
        self.num_threads = num_threads
        self.sampler = sampler

    def _put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        pass

    def _get_tensor(self, attr: TensorAttr) -> FeatureTensorType | None:
        return self.sampler.get_tensor(attr)

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        pass

    def get_all_tensor_attrs(self) -> list[TensorAttr]:
        """Return all TensorAttr from the table nodes."""
        return self.client.get_node_groups_and_features()
