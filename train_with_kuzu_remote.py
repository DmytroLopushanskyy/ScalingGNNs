import multiprocessing as mp

import kuzu
import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MLP, BatchNorm, SAGEConv

import matplotlib.pyplot as plt

NUM_EPOCHS = 50
LOADER_BATCH_SIZE = 128

print('Batch size:', LOADER_BATCH_SIZE)
print('Number of epochs:', NUM_EPOCHS)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

########################################################################
# The below code sets up the remote backend of Kùzu for PyG.
# Please refer to: https://kuzudb.com/docs/client-apis/python-api/overview.html
# for how to use the Python API of Kùzu.
########################################################################

# The buffer pool size of Kùzu is set to 2GB (default is 80% of available memory)
KUZU_BM_SIZE = 2 * 1024**3

# Create Kùzu database:
db = kuzu.Database('cora', buffer_pool_size=KUZU_BM_SIZE)
conn = kuzu.Connection(db)

# Get remote backend for PyG:
feature_store, graph_store = db.get_torch_geometric_remote_backend(
    mp.cpu_count())

count_result = conn.execute('MATCH (p:paper) RETURN count(*);')
num_papers = count_result.get_next()[0]

train_count = int(0.6 * num_papers)
test_count = num_papers - train_count
train_ids, test_ids = torch.utils.data.random_split(
    range(num_papers), (train_count, test_count),
    generator=torch.Generator().manual_seed(42)
)
train_mask = torch.zeros(num_papers, dtype=torch.bool)
test_mask = torch.zeros(num_papers, dtype=torch.bool)
train_mask.index_fill_(0, torch.LongTensor(train_ids), True)
test_mask.index_fill_(0, torch.LongTensor(test_ids), True)

# Plug the graph store and feature store into the `NeighborLoader`.
# Note that `filter_per_worker` is set to `False`. This is because the Kùzu
# database is already using multi-threading to scan the features in parallel
# and the database object is not fork-safe.
# loader = NeighborLoader(
#     data=(feature_store, graph_store),
#     num_neighbors={('paper', 'cites', 'paper'): [12, 12, 12]},
#     batch_size=LOADER_BATCH_SIZE,
#     input_nodes=('paper', input_nodes),
#     num_workers=4,
#     filter_per_worker=False,
# )

loader = NeighborLoader(
    data=(feature_store, graph_store),
    num_neighbors={key.edge_type: [30] * 2 for key in graph_store.get_all_edge_attrs()},
    batch_size=LOADER_BATCH_SIZE,  # Use a batch size of 128 for sampling training nodes
    input_nodes=('paper', train_mask),
    filter_per_worker=False,
    # num_workers=4,
)

print(graph_store.get_all_edge_attrs())

data = Planetoid('./cora/data', name='Cora')[0]
loader = NeighborLoader(
    data=data,
    num_neighbors=[30] * 2,  # Sample 30 neighbors for each node for 2 iterations
    batch_size=LOADER_BATCH_SIZE,  # Use a batch size of 128 for sampling training nodes
    input_nodes=data.train_mask,
    num_workers=0,
    filter_per_worker=False
)

print(next(iter(loader)))

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


model = GraphSAGE(in_channels=1433, hidden_channels=1024, out_channels=172,
                  num_layers=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
losses = []

for epoch in range(1, NUM_EPOCHS + 1):
    total_loss = total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        batch_size = batch.batch_size

        optimizer.zero_grad()
        out = model(
            batch.x,
            batch.edge_index,
        )[:batch_size]
        y = batch.y[:batch_size].long().view(-1)
        loss = F.cross_entropy(out, y)

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * y.numel()
        total_examples += y.numel()

    avg_loss = total_loss / total_examples
    losses.append(avg_loss)

    print(f'Epoch: {epoch:02d}, Loss: {total_loss / total_examples:.4f}')

# Plot and save the loss graph
# plt.figure()
# plt.plot(range(1, NUM_EPOCHS + 1), losses, label='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss (No Remote Backend)')
# plt.legend()
# plt.savefig('loss-no-backend.png')