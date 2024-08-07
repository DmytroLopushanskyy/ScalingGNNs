from __future__ import annotations

import multiprocessing

import numpy as np
import torch
from torch_geometric.typing import FeatureTensorType
from torch_geometric.data.feature_store import TensorAttr, FeatureStore
from neo4j_remote_backend.client import Neo4jClient

from src.graph_sampler import sampler


class Neo4jFeatureStore(FeatureStore):
    def __init__(self, client: Neo4jClient | None = None, num_threads: int | None = None):
        print("Neo4jFeatureStore __init__", num_threads)
        super().__init__()
        if num_threads is None:
            num_threads = multiprocessing.cpu_count()
        if client is None:
            client = Neo4jClient(num_threads)

        self.__client = client
        self.num_threads = num_threads
        self.sampler = sampler

    def _put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        print("_put_tensor")
        print("tensor", tensor)
        print("attr", attr)
        pass

    def _get_tensor(self, attr: TensorAttr) -> FeatureTensorType | None:
        # print("_get_tensor")
        # print("attr", attr)
        # print("index size", attr.index.size())
        # print(self.__client.get_tensor_by_query(attr))
        # print("GET TENSOR", attr)
        # return self.__client.get_tensor_by_query(attr) # torch.tensor([168,169,182,239,245,252,259,268,276,285,291,296,312,325,355,363,167,2437,2438,1994,2706,183,997,65,619,1069,1274,1376,1759,1909,2021,2182,2418,782,1162,711,973,1973,2485,117,2537,1740,2451,1463,1695,2133,502,1999,1692,732,2685,1610,1722,787,1848,1668,1003,1056,2482,1986,165,1473,2707,1837,543,771,1156,1293,1628,2419,2232,493,569,1358,211,1131,1171,2305,55,525,858,415,2072,2172,2180,1139,2233,57,572,1510,414,496,539,644,1285,2111,2112,2113,277,176,630,671,1142,682,2016,454,2153,504,779,1333,1482,490,1637,822,1974,1975,1976,1370,1592,1792,1851,2050,1035,1072,1627,1725,2389,2390,915,2392,367,752,1801,1804,2391,2452,555,2131,2132,434,2054,279,818,827,1469,1472,240,2186,2187,172,2346,2347,2348,756,263,364,1756,1718,244,565,687,2556,2557,2558,289,438,1013,1224,1652,1527,598,1373,1687,2607], dtype=torch.int64)
        return self.sampler.get_tensor(attr)

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        print("_remove_tensor")
        print("attr", attr)
        pass

    def get_all_tensor_attrs(self) -> list[TensorAttr]:
        """Return all TensorAttr from the table nodes."""
        # [TensorAttr(group_name='paper', attr_name='id', index=<_FieldStatus.UNSET: None>), TensorAttr(group_name='paper', attr_name='x', index=<_FieldStatus.UNSET: None>), TensorAttr(group_name='paper', attr_name='y', index=<_FieldStatus.UNSET: None>)]
        # print('get_all_tensor_attrs', self.__client.get_node_groups_and_features())
        # return [TensorAttr('paper', 'id', None), TensorAttr('paper', 'x', None), TensorAttr('paper', 'y', None)]
        return self.__client.get_node_groups_and_features()
