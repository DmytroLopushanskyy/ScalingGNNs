from __future__ import annotations

import multiprocessing
import numpy as np
import torch
from torch_geometric.typing import FeatureTensorType
from torch_geometric.data.feature_store import TensorAttr, FeatureStore


class Neo4jFeatureStore(FeatureStore):
    def __init__(self, num_threads: int | None = None):
        super().__init__()
        if num_threads is None:
            num_threads = multiprocessing.cpu_count()
        self.num_threads = num_threads

    def _put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        raise NotImplementedError

    def _get_tensor(self, attr: TensorAttr) -> FeatureTensorType | None:
        raise NotImplementedError

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        raise NotImplementedError

    def get_all_tensor_attrs(self) -> list[TensorAttr]:
        """Return all TensorAttr from the table nodes."""
        raise NotImplementedError
