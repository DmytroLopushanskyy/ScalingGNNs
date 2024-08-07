import kuzu
import torch
from torch_geometric.loader import NeighborLoader
from multiprocessing import cpu_count

from src.graph_sage import GraphSAGE
# from src.model_v2 import CoraNodeClassification
from src.utils import model_params, device, train_mask, test_mask
from src.train_test import train, test

import numpy as np
import pandas as pd

torch.manual_seed(12345)

def main():
    db = kuzu.Database('data/kuzu-4')
    feature_store, graph_store = db.get_torch_geometric_remote_backend(cpu_count())

    train_path = "/data/coml-intersection-joins/kebl7757/ScalingGNNs/papers_100M_classification/dataset/ogbn_papers100M/split/time/train.csv.gz"
    train_df = pd.read_csv(train_path, compression='gzip', header=None)
    train_nodes = torch.tensor(train_df[0].values, dtype=torch.long)
    test_path = "/data/coml-intersection-joins/kebl7757/ScalingGNNs/papers_100M_classification/dataset/ogbn_papers100M/split/time/test.csv.gz"
    test_df = pd.read_csv(test_path, compression='gzip', header=None)
    test_nodes = torch.tensor(test_df[0].values, dtype=torch.long)

    print("train_nodes", len(train_nodes))
    print("test_nodes", len(test_nodes))

    loader_params = {
        "num_neighbors": [12,12],
        "batch_size": 1152,
        "num_workers": 4,
        "filter_per_worker": False
    }

    train_loader = NeighborLoader(**loader_params, data=(feature_store, graph_store), input_nodes=('paper', train_nodes))
    test_loader = NeighborLoader(**loader_params, data=(feature_store, graph_store), input_nodes=('paper', test_nodes))

    # model = CoraNodeClassification().to(device)
    model = GraphSAGE(in_channels=128, hidden_channels=1024, out_channels=172, num_layers=3).to(device)

    mode = 'kuzu'
    train(model, train_loader, test_loader, model_params, mode)
    test_accuracy = test(model, test_loader, mode)
    print("Final test accuracy:", test_accuracy)

if __name__ == '__main__':
    main()