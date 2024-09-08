import builtins
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader
from memory_profiler import memory_usage

from src.graph_sage import GraphSAGE
from src.train_test import train, test
from torch_geometric.data import HeteroData

import numpy as np
import pandas as pd
from datetime import datetime

from ogb.nodeproppred import PygNodePropPredDataset

torch.manual_seed(12345)

def main():
    print("#"*40 + f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    builtins.input = lambda _: 'N'
    dataset = PygNodePropPredDataset(name='ogbn-products', root='/data/coml-intersection-joins/kebl7757/ScalingGNNs/distributed_training/distributed-neo4j/data/dataset/ogbn-products/')
    # split_idx = dataset.get_idx_split()
    
    hetero_data = HeteroData()
    data = dataset[0]
    hetero_data['PRODUCT'].features = data.x
    hetero_data['PRODUCT'].label = data.y
    hetero_data['PRODUCT'].num_nodes = data.num_nodes
    hetero_data['PRODUCT', 'LINK', 'PRODUCT'].edge_index = data.edge_index
    
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

    train_loader = NeighborLoader(**loader_params, data=hetero_data, input_nodes=('PRODUCT', train_nodes))
    test_loader = NeighborLoader(**loader_params, data=hetero_data, input_nodes=('PRODUCT', test_nodes))

    model = GraphSAGE(in_channels=100, hidden_channels=256, out_channels=47, num_layers=3)

    train_mem_usage = memory_usage((train, (model, train_loader, test_loader, train_params)))
    print(f"Maximum memory usage during training: {max(train_mem_usage):.2f} MB")
    test_mem_usage = memory_usage((test, (model, test_loader)))
    print(f"Maximum memory usage during testing: {max(test_mem_usage):.2f} MB")


if __name__ == '__main__':
    main()