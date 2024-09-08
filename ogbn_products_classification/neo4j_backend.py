import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.loader import NeighborLoader
from memory_profiler import memory_usage

from src.graph_sage import GraphSAGE
from src.train_test import train, test

from src.client import Neo4jClient
from src.feature_store import Neo4jFeatureStore
from src.graph_store import Neo4jGraphStore
from src.graph_sampler import GraphSampler

import pandas as pd

torch.manual_seed(12345)


def main():
    db = Neo4jClient()
    sampler = GraphSampler()
    feature_store = Neo4jFeatureStore(db, sampler)
    graph_store = Neo4jGraphStore(db)

    train_path = "/data/coml-intersection-joins/kebl7757/ScalingGNNs/distributed_training/distributed-neo4j/data/dataset/ogbn-products/ogbn_products/split/sales_ranking/train.csv.gz"
    train_df = pd.read_csv(train_path, compression='gzip', header=None)
    train_idx = torch.tensor(train_df[0].values, dtype=torch.long)
    train_nodes = train_idx.tolist()

    test_path = "/data/coml-intersection-joins/kebl7757/ScalingGNNs/distributed_training/distributed-neo4j/data/dataset/ogbn-products/ogbn_products/split/sales_ranking/test.csv.gz"
    test_df = pd.read_csv(test_path, compression='gzip', header=None)
    test_idx = torch.tensor(test_df[0].values, dtype=torch.long)
    test_nodes = test_idx.tolist()

    loader_params = {
        "num_neighbors": [15,10,5],
        "batch_size": 1024,
        "num_workers": 0,
        "filter_per_worker": False
    }
    train_params = {
        "batch_size": 1024, 
        "num_epochs": 100,
        "learning_rate": 0.0004,
        "weight_decay": 5e-4
    }

    train_loader = NeighborLoader(**loader_params, data=(feature_store, graph_store), input_nodes=('PRODUCT', train_nodes))
    test_loader = NeighborLoader(**loader_params, data=(feature_store, graph_store), input_nodes=('PRODUCT', test_nodes))

    print("NeighborLoaders initialized")
    model = GraphSAGE(in_channels=100, hidden_channels=256, out_channels=47, num_layers=3)

    mode = 'neo4j'
    train_mem_usage = memory_usage((train, (model, train_loader, test_loader, train_params, mode)))
    print(f"Maximum memory usage during training: {max(train_mem_usage):.2f} MB")
    test_mem_usage = memory_usage((test, (model, test_loader, mode)))
    print(f"Maximum memory usage during testing: {max(test_mem_usage):.2f} MB")


if __name__ == '__main__':
    main()