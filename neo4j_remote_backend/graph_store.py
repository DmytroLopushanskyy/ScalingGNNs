from __future__ import annotations

import multiprocessing
import numpy as np
import torch
from torch_geometric.data.graph_store import GraphStore


class Neo4jGraphStore(GraphStore):
    def __init__(self, num_threads: int | None = None):
        super().__init__()
        if num_threads is None:
            num_threads = multiprocessing.cpu_count()
        self.num_threads = num_threads
