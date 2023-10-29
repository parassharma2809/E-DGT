import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class Graph(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, dropout=0.1):
        super(Graph, self).__init__()
        self.fc_1 = nn.Linear(in_dim, hidden_dim)
        self.fc_2 = nn.Linear(in_dim, hidden_dim)

        hidden_dim = out_dim if n_layers == 1 else hidden_dim
        self.layers = nn.ModuleList([
            GraphConvolution(in_dim, hidden_dim)
        ])

        for i in range(n_layers - 1):
            temp = out_dim if i == n_layers - 2 else hidden_dim
            self.layers.append(GraphConvolution(hidden_dim, temp))

        self.dropout = dropout

    def forward(self, X, A):
        for layer in self.layers:
            X = F.relu(layer(X, A))
            X = F.dropout(X, self.dropout, training=self.training)
        return X


class GraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, skip=True):
        super(GraphConvolution, self).__init__()
        self.skip = skip
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = Parameter(torch.Tensor(in_dim, out_dim))
        self.bias = Parameter(torch.Tensor(out_dim))

    def forward(self, x, adj):
        support = torch.bmm(x, self.weight.unsqueeze(0).expand(x.shape[0], -1, -1))
        output = torch.bmm(adj, support)
        output += self.bias.unsqueeze(0).expand(x.shape[0], -1, -1)
        if self.skip:
            output += support
        return output
