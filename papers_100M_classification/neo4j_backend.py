import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.loader import NeighborLoader
from memory_profiler import memory_usage

from src.graph_sage import GraphSAGE
# from src.model_v2 import CoraNodeClassification
from src.utils import model_params, device
from src.train_test import train, test

from neo4j_remote_backend.client import Neo4jClient
from neo4j_remote_backend.feature_store import Neo4jFeatureStore
from neo4j_remote_backend.graph_store import Neo4jGraphStore
from src.graph_sampler import GraphSampler

import pandas as pd

torch.manual_seed(12345)


def main():
    db = Neo4jClient()
    sampler = GraphSampler()
    feature_store = Neo4jFeatureStore(db, sampler)
    graph_store = Neo4jGraphStore(db)

    train_path = "/data/coml-intersection-joins/kebl7757/ScalingGNNs/papers_100M_classification/dataset/ogbn_papers100M/split/time/train.csv.gz"
    train_df = pd.read_csv(train_path, compression='gzip', header=None)
    train_idx = torch.tensor(train_df[0].values, dtype=torch.long)
    train_nodes = train_idx.tolist()

    test_path = "/data/coml-intersection-joins/kebl7757/ScalingGNNs/papers_100M_classification/dataset/ogbn_papers100M/split/time/test.csv.gz"
    test_df = pd.read_csv(test_path, compression='gzip', header=None)
    test_idx = torch.tensor(test_df[0].values, dtype=torch.long)
    test_nodes = test_idx.tolist()

    loader_params = {
        "num_neighbors": [2],
        "batch_size": 1024,
        "num_workers": 0,
        "filter_per_worker": False
    }

    train_loader = NeighborLoader(**loader_params, data=(feature_store, graph_store), input_nodes=('PAPER', train_nodes))
    test_loader = NeighborLoader(**loader_params, data=(feature_store, graph_store), input_nodes=('PAPER', test_nodes))

    # model = CoraNodeClassification().to(device)
    model = GraphSAGE(in_channels=128, hidden_channels=1024, out_channels=172, num_layers=3).to(device)

    mode = 'neo4j'
    train_mem_usage = memory_usage((train, (model, train_loader, test_loader, model_params, mode)))
    print(f"Maximum memory usage during training: {max(train_mem_usage):.2f} MB")
    test_mem_usage = memory_usage((test, (model, test_loader, mode)))
    print(f"Maximum memory usage during testing: {max(test_mem_usage):.2f} MB")


if __name__ == '__main__':
    main()