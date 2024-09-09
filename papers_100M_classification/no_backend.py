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
from datetime import datetime
import matplotlib.pyplot as plt

from ogb.nodeproppred import PygNodePropPredDataset


def main():
    print("#"*40 + f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    builtins.input = lambda _: 'N'
    dataset = PygNodePropPredDataset(name='ogbn-papers100M')
    
    hetero_data = HeteroData()
    data = dataset[0]
    hetero_data['paper'].x = data.x
    hetero_data['paper'].y = data.y
    hetero_data['paper'].num_nodes = data.num_nodes
    hetero_data['paper', 'cites', 'paper'].edge_index = data.edge_index
    
    train_path = "/data/project/anonym/ScalingGNNs/papers_100M_classification/dataset/ogbn_papers100M/split/time/train.csv.gz"
    train_df = pd.read_csv(train_path, compression='gzip', header=None)
    train_idx = torch.tensor(train_df[0].values, dtype=torch.long)
    train_nodes = train_idx.tolist()

    test_path = "/data/project/anonym/ScalingGNNs/papers_100M_classification/dataset/ogbn_papers100M/split/time/test.csv.gz"
    test_df = pd.read_csv(test_path, compression='gzip', header=None)
    test_idx = torch.tensor(test_df[0].values, dtype=torch.long)
    test_nodes = test_idx.tolist()

    loader_params = {
        "num_neighbors": [12,12],
        "batch_size": 4096,
        "num_workers": 0,
        "filter_per_worker": False
    }

    train_loader = NeighborLoader(**loader_params, data=hetero_data, input_nodes=('paper', train_nodes))
    test_loader = NeighborLoader(**loader_params, data=hetero_data, input_nodes=('paper', test_nodes))

    # model = CoraNodeClassification().to(device)
    model = GraphSAGE(in_channels=128, hidden_channels=1024, out_channels=172, num_layers=3).to(device)

    interval = 0.1  # in seconds
    train_mem_usage = memory_usage((train, (model, train_loader, test_loader, model_params)), interval=interval)
    print(f"Maximum memory usage during training: {max(train_mem_usage):.2f} MB")
    test_mem_usage = memory_usage((test, (model, test_loader)))
    print(f"Maximum memory usage during testing: {max(test_mem_usage):.2f} MB")
    
    time_in_seconds = [i * interval for i in range(len(train_mem_usage))]
    memory_log = pd.DataFrame({'Time (s)': time_in_seconds, 'Memory Usage (MiB)': train_mem_usage })
    memory_log.to_csv('memory_usage_log.csv', index=False)
    plt.figure(figsize=(10, 6))
    plt.plot(memory_log['Time (s)'], memory_log['Memory Usage (MiB)'], label='Memory Usage')
    plt.xlabel('Time (s)')
    plt.ylabel('Memory Usage (MiB)')
    plt.title('Memory Usage Over Time During Training')
    plt.legend()
    plt.savefig('memory_usage_log.png')


if __name__ == '__main__':
    main()