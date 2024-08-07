import builtins
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader
from memory_profiler import memory_usage

from src.graph_sage import GraphSAGE
# from src.model_v2 import CoraNodeClassification
from src.utils import model_params, device, train_mask, test_mask
from src.train_test import train, test
from torch_geometric.data import HeteroData

import numpy as np
import pandas as pd

from ogb.nodeproppred import PygNodePropPredDataset

torch.manual_seed(12345)

def main():
    builtins.input = lambda _: 'N'
    dataset = PygNodePropPredDataset(name='ogbn-papers100M')
    # split_idx = dataset.get_idx_split()
    
    hetero_data = HeteroData()
    data = dataset[0]
    hetero_data['paper'].x = data.x
    hetero_data['paper'].y = data.y
    hetero_data['paper'].num_nodes = data.num_nodes
    hetero_data['paper', 'cites', 'paper'].edge_index = data.edge_index
    
    # train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    train_path = "/data/coml-intersection-joins/kebl7757/ScalingGNNs/papers_100M_classification/dataset/ogbn_papers100M/split/time/train.csv.gz"
    train_df = pd.read_csv(train_path, compression='gzip', header=None)
    train_idx = torch.tensor(train_df[0].values, dtype=torch.long)
    train_nodes = train_idx.tolist()

    test_path = "/data/coml-intersection-joins/kebl7757/ScalingGNNs/papers_100M_classification/dataset/ogbn_papers100M/split/time/test.csv.gz"
    test_df = pd.read_csv(test_path, compression='gzip', header=None)
    test_idx = torch.tensor(test_df[0].values, dtype=torch.long)
    test_nodes = test_idx.tolist()

    loader_params = {
        "num_neighbors": [12,12,12],
        "batch_size": 1152,
        "num_workers": 4,
        "filter_per_worker": False
    }

    train_loader = NeighborLoader(**loader_params, data=hetero_data, input_nodes=('paper', train_nodes))
    test_loader = NeighborLoader(**loader_params, data=hetero_data, input_nodes=('paper', test_nodes))

    # model = CoraNodeClassification().to(device)
    model = GraphSAGE(in_channels=128, hidden_channels=1024, out_channels=172, num_layers=3).to(device)

    train_mem_usage = memory_usage((train, (model, train_loader, test_loader, model_params)))
    print(f"Maximum memory usage during training: {max(train_mem_usage):.2f} MB")
    test_mem_usage = memory_usage((test, (model, test_loader)))
    print(f"Maximum memory usage during testing: {max(test_mem_usage):.2f} MB")


if __name__ == '__main__':
    main()