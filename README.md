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

## Processing large datasets

Loading a large csv file into Kuzu requires significant memory usage. For example, a job loading 30GB of edges into Kuzu with 16 GB memory limited failed with out-of-memory error.

To avoid loading a big file into Kuzu in one go, I decided to split it into 1GB-sized partitions. This was done using the following command:
```
split -b 1G /data/coml-intersection-joins/kebl7757/ScalingGNNsapers_100M_classification/data/papers100M/processed/edge_index.csv /data/coml-intersection-joins/kebl7757/ScalingGNNs/papers_100M_classification/data/papers100M/processed/split_
```

However, this may cause artifacts at the beginning and end of each splitted file as a single CSV line might be split broken between two files. To fix this, I've created a bash script `print_first_last_lines.sh` that prints out the first and last line that can be merged manually.

Finally, the `remove_first_last_lines.sh` needs to be run to delete the first and last line from each csv file. The manually corrected lines can then be inserted in a seperate csv file.


### Env installation

```
conda install pytorch==2.3.0 torchvision torchaudio cpuonly -c pytorch
conda install -c conda-forge cmake
conda install -c conda-forge gcc=9 gxx=9
pip install git+https://github.com/pyg-team/pyg-lib.git@0.4.0
```