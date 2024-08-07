import multiprocessing as mp
import os.path as osp
from datetime import datetime

import kuzu
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MLP, BatchNorm, SAGEConv

torch.set_printoptions(threshold=100000)

NUM_EPOCHS = 1
LOADER_BATCH_SIZE = 1024

print('Batch size:', LOADER_BATCH_SIZE)
print('Number of epochs:', NUM_EPOCHS)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Load the train set:
train_path = "/data/coml-intersection-joins/kebl7757/ScalingGNNs/papers_100M_classification/dataset/ogbn_papers100M/split/time/train.csv.gz"
train_df = pd.read_csv(
    train_path,
    compression='gzip',
    header=None,
)
input_nodes = torch.tensor(train_df[0].values, dtype=torch.long)
input_node_list = input_nodes.tolist()
input_node_set = set(input_node_list)

########################################################################
# The below code sets up the remote backend of Kùzu for PyG.
# Please refer to: https://kuzudb.com/docs/client-apis/python-api/overview.html
# for how to use the Python API of Kùzu.
########################################################################

# The buffer pool size of Kùzu is set to 40GB. You can change it to a smaller
# value if you have less memory.
KUZU_BM_SIZE = 40 * 1024**3

# Create Kùzu database:
db = kuzu.Database(database_path='data/kuzu-4', buffer_pool_size=KUZU_BM_SIZE)

# Get remote backend for PyG:
feature_store, graph_store = db.get_torch_geometric_remote_backend(
    mp.cpu_count())

# Plug the graph store and feature store into the `NeighborLoader`.
# Note that `filter_per_worker` is set to `False`. This is because the Kùzu
# database is already using multi-threading to scan the features in parallel
# and the database object is not fork-safe.
print("loader")

loader = NeighborLoader(
    data=(feature_store, graph_store),
    num_neighbors=[12],
    batch_size=LOADER_BATCH_SIZE,
    input_nodes=('paper', input_nodes),
    num_workers=4,
    filter_per_worker=False
)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=0.2):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(BatchNorm(hidden_channels))
        for i in range(1, num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(BatchNorm(hidden_channels))

        self.mlp = MLP(
            in_channels=in_channels + num_layers * hidden_channels,
            hidden_channels=2 * out_channels,
            out_channels=out_channels,
            num_layers=2,
            norm='batch_norm',
            act='leaky_relu',
        )

        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        xs = [x]
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = x.relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        return self.mlp(torch.cat(xs, dim=-1))


model = GraphSAGE(in_channels=128, hidden_channels=1024, out_channels=172,
                  num_layers=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(1, NUM_EPOCHS + 1):
    total_loss = total_examples = 0
    # print("before batch")
    for batch in tqdm(loader):
        print("batch!", datetime.now().strftime("%A, %B %d, %Y %I:%M:%S %p"))
        batch = batch.to(device)
        batch_size = batch['paper'].batch_size

        batch_node_indices = batch['paper'].n_id.cpu().numpy()
        batch_labels = batch['paper'].y[:batch_size].cpu().numpy()

        # print(f"Batch node indices: {batch_node_indices}")
        # print(f"Batch labels: {batch_labels}")

        # Compare with input nodes
        nodes_in_input = [node for node in batch_node_indices if node in input_node_set]
        nodes_not_in_input = [node for node in batch_node_indices if node not in input_node_set]

        # print(f"Nodes in input_nodes: {nodes_in_input}")
        # print(f"Nodes not in input_nodes: {nodes_not_in_input}")

        optimizer.zero_grad()
        # print("model before")
        out = model(
            batch['paper'].x,
            batch['paper', 'cites', 'paper'].edge_index,
        )[:batch_size]
        # print("model after")
        y = batch['paper'].y[:batch_size].long().view(-1)
        valid_mask = y != -9223372036854775808  # Mask for valid labels
        
        # Only use valid labels for loss calculation
        if valid_mask.sum() > 0:
            valid_out = out[valid_mask]
            valid_y = y[valid_mask]

            loss = F.cross_entropy(valid_out, valid_y)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * valid_y.numel()
            total_examples += valid_y.numel()

    print(f'Epoch: {epoch:02d}, Loss: {total_loss / total_examples:.4f}')