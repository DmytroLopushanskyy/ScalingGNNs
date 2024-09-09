import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.loader import NeighborLoader
from memory_profiler import memory_usage

from src.model_v2 import CoraNodeClassification
from src.utils import model_params, device, test_mask
from src.train_test import train, test

from src.utils import loader_params, train_mask
from neo4j_remote_backend.client_for_cora import Neo4jClient
from neo4j_remote_backend.feature_store import Neo4jFeatureStore
from neo4j_remote_backend.graph_store import Neo4jGraphStore

torch.manual_seed(12345)


def main():
    db = Neo4jClient()
    feature_store = Neo4jFeatureStore(db)
    graph_store = Neo4jGraphStore(db)

    train_loader = NeighborLoader(**loader_params, data=(feature_store, graph_store), input_nodes=('Paper', train_mask))
    test_loader = NeighborLoader(**loader_params, data=(feature_store, graph_store), input_nodes=('Paper', test_mask))

    model = CoraNodeClassification().to(device)

    mode = 'neo4j'
    train_mem_usage = memory_usage((train, (model, train_loader, test_loader, model_params, mode)))
    print(f"Maximum memory usage during training: {max(train_mem_usage):.2f} MB")
    test_mem_usage = memory_usage((test, (model, test_loader, mode)))
    print(f"Maximum memory usage during testing: {max(test_mem_usage):.2f} MB")


if __name__ == '__main__':
    main()