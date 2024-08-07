# Some of the code below reused from https://github.com/russellizadi/ssp
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class CRD(torch.nn.Module):
    def __init__(self, d_in, d_out, p):
        super(CRD, self).__init__()
        self.conv = GCNConv(d_in, d_out)
        self.p = p

    def forward(self, x, edge_index):
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        return x


class CLS(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(CLS, self).__init__()
        self.conv = GCNConv(d_in, d_out)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


class CoraNodeClassification(torch.nn.Module):
    def __init__(self):
        super(CoraNodeClassification, self).__init__()
        self.crd = CRD(128, 16, 0.5)
        self.cls = CLS(16, 172)

    def forward(self, x, edge_index):
        x = self.crd(x, edge_index)
        x = self.cls(x, edge_index)
        return x
