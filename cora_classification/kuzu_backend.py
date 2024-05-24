import kuzu
import torch
from torch_geometric.loader import NeighborLoader
from multiprocessing import cpu_count

from src.model_v2 import CoraNodeClassification
from src.utils import model_params, loader_params, device, train_mask, test_mask
from src.train_test import train, test

KUZU_BM_SIZE = 2 * 1024 ** 3  # Buffer pool size of 2GB (default is 80% of memory)
torch.manual_seed(12345)

def main():
    db = kuzu.Database('data/kuzu', buffer_pool_size=KUZU_BM_SIZE)
    feature_store, graph_store = db.get_torch_geometric_remote_backend(cpu_count())

    train_loader = NeighborLoader(**loader_params, data=(feature_store, graph_store), input_nodes=('paper', train_mask))
    test_loader = NeighborLoader(**loader_params, data=(feature_store, graph_store), input_nodes=('paper', test_mask))

    model = CoraNodeClassification().to(device)

    mode = 'kuzu'
    train(model, train_loader, test_loader, model_params, mode)
    test_accuracy = test(model, test_loader, mode)
    print("Final test accuracy:", test_accuracy)


if __name__ == '__main__':
    main()