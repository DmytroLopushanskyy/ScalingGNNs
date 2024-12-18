import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader
from memory_profiler import memory_usage

from src.model_v2 import CoraNodeClassification
from src.utils import model_params, loader_params, device, train_mask, test_mask
from src.train_test import train, test

torch.manual_seed(12345)

def main():
    data = Planetoid('./data', name='Cora')[0]
    train_loader = NeighborLoader(**loader_params, data=data, input_nodes=train_mask)
    test_loader = NeighborLoader(**loader_params, data=data, input_nodes=test_mask)

    model = CoraNodeClassification().to(device)

    train_mem_usage = memory_usage((train, (model, train_loader, test_loader, model_params)))
    print(f"Maximum memory usage during training: {max(train_mem_usage):.2f} MB")
    test_mem_usage = memory_usage((test, (model, test_loader)))
    print(f"Maximum memory usage during testing: {max(test_mem_usage):.2f} MB")


if __name__ == '__main__':
    main()