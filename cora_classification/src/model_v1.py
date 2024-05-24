# Some of the code below reused from Oxford GRL Practical 2.
import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GNNLayer, self).__init__()
        self.W_self = nn.Linear(input_dim, output_dim, bias=False)
        self.W_neigh = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, node_feats, adj_matrix):
        h_self = self.W_self(node_feats)
        h_neigh = self.W_neigh(torch.matmul(adj_matrix, node_feats))
        return h_self + h_neigh


class GNNModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, act_fn=nn.ReLU):
        super(GNNModule, self).__init__()
        self.layers = nn.ModuleList()
        self.act_fn = act_fn()

        # Create the first layer
        self.layers.append(GNNLayer(input_dim, hidden_dim))
        # Create `num_layers-1` layers where each layer takes the hidden dimension as input and output
        for _ in range(num_layers - 1):
            self.layers.append(GNNLayer(hidden_dim, hidden_dim))
        # Output layer
        self.out_layer = GNNLayer(hidden_dim, output_dim)

    def forward(self, x, adj_matrix):
        for layer in self.layers:
            x = self.act_fn(layer(x, adj_matrix))
        return self.out_layer(x, adj_matrix)


class MLPModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, act_fn=nn.ReLU):
        super(MLPModule, self).__init__()
        self.layers = nn.ModuleList()
        self.act_fn = act_fn()

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act_fn(layer(x))
        return self.layers[-1](x)


class CoraNodeClassification(nn.Module):
    def __init__(self, input_dim, num_gnn_layers, num_mlp_layers, hidden_features, num_classes, gnn_act_fn=nn.ReLU,
                 mlp_act_fn=nn.ReLU):
        super(CoraNodeClassification, self).__init__()
        self.gnn = GNNModule(input_dim, hidden_features, hidden_features, num_gnn_layers, gnn_act_fn)
        self.mlp = MLPModule(hidden_features, hidden_features, num_classes, num_mlp_layers, mlp_act_fn)

    def forward(self, node_feats, adj_matrix):
        node_feats = self.gnn(node_feats, adj_matrix)
        node_feats = self.mlp(node_feats)

        return F.log_softmax(node_feats, dim=1)
