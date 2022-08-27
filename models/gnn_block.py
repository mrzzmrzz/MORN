import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as pyg
import torch_geometric.nn as gnn


class GNNBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.conv = gnn.GINEConv(self.mlp)

    def forward(self, node_feat, edge_index, edge_attr):
        x = self.conv.forward(node_feat, edge_index, edge_attr)
        x = F.relu(x)
        return x
