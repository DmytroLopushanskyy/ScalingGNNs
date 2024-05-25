# Scaling Up GNNs with Remote Backends

## NeighborLoader

Intuitively, NeighborLoader performs the following steps:
1. The loader picks the next set of nodes based on the batch size.
2. It samples the specified number of neighbors for these nodes.
3. It constructs a subgraph with these nodes and their sampled neighbors.
4. The subgraph is then used for a training iteration of your GNN.

Actual sampling happens either through pyg-lib or torch-sparse. See source code below:
```python
out = torch.ops.torch_sparse.hetero_neighbor_sample(
    self.node_types,
    self.edge_types,
    self.colptr_dict,
    self.row_dict,
    seed,  # seed_dict
    self.num_neighbors.get_mapped_values(self.edge_types),
    self.num_neighbors.num_hops,
    self.replace,
    self.subgraph_type != SubgraphType.induced,
)
```