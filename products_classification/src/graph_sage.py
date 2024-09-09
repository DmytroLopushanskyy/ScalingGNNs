import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP, BatchNorm, SAGEConv


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