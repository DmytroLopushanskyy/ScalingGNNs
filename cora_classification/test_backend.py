import kuzu
import torch
from torch_geometric.loader import NeighborLoader
from multiprocessing import cpu_count

from src.utils import loader_params, train_mask

KUZU_BM_SIZE = 2 * 1024 ** 3  # Buffer pool size of 2GB (default is 80% of memory)
torch.manual_seed(12345)

def main():
    db = kuzu.Database('data/kuzu', buffer_pool_size=KUZU_BM_SIZE)
    feature_store, graph_store = db.get_torch_geometric_remote_backend(cpu_count())

    loader = NeighborLoader(**loader_params, data=(feature_store, graph_store), input_nodes=('paper', train_mask))

    for batch in loader:
        for row in batch['paper'].x:
            lst = (row != 0).nonzero(as_tuple=False).squeeze().tolist()
            print(lst)
        break


if __name__ == '__main__':
    main()