import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basic_block import MLP


class CLFUModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim_in = config["dim_in"]
        self.dim_hid = config["dim_hid"]
        mlp_config = config["mlp_config"]

        self.lin = nn.Linear(self.dim_in, self.dim_hid)

        self.mol_fusion = nn.Sequential(
            nn.Linear(self.dim_hid * 2, self.dim_hid),
            nn.ReLU(True),
            nn.LayerNorm(self.dim_hid),
            nn.Linear(self.dim_hid, self.dim_hid),
        )

        self.classifier = MLP(mlp_config)

    def forward(self, graph_repr, seq_repr):
        batch_num = graph_repr.size(0)

        graph_repr = self.lin(graph_repr)
        seq_repr = self.lin(seq_repr)

        cross_graph_to_seq_repr = torch.einsum("ik,jk->ij", graph_repr, seq_repr)
        cross_seq_to_graph_repr = torch.einsum("ik,jk->ij", seq_repr, graph_repr)

        abs_graph = torch.norm(graph_repr, dim=1)
        abs_seq = torch.norm(seq_repr, dim=1)

        abs_graph_to_seq = torch.einsum("i,j->ij", abs_graph, abs_seq)
        abs_seq_to_graph = torch.einsum("i,j->ij", abs_seq, abs_graph)

        sim_graph_to_seq_matrix = cross_graph_to_seq_repr / abs_graph_to_seq
        sim_graph_to_seq_matrix = torch.exp(sim_graph_to_seq_matrix / 0.1)
        pos_sim_graph_to_seq = sim_graph_to_seq_matrix[
            range(batch_num), range(batch_num)
        ]
        loss_graph_to_seq = pos_sim_graph_to_seq / (
            sim_graph_to_seq_matrix.sum(dim=1) - pos_sim_graph_to_seq
        )
        loss_graph_to_seq = -torch.log(loss_graph_to_seq).mean()

        sim_seq_to_graph_matrix = cross_seq_to_graph_repr / abs_seq_to_graph
        sim_seq_to_graph_matrix = torch.exp(sim_seq_to_graph_matrix / 0.1)
        pos_sim_seq_to_graph = sim_seq_to_graph_matrix[
            range(batch_num), range(batch_num)
        ]
        loss_seq_to_graph = pos_sim_seq_to_graph / (
            sim_seq_to_graph_matrix.sum(dim=1) - pos_sim_seq_to_graph
        )
        loss_seq_to_graph = -torch.log(loss_seq_to_graph).mean()

        loss = 0.5 * (loss_graph_to_seq + loss_seq_to_graph)

        # fuse_repr = graph_repr + 0.05 * seq_repr

        fuse_repr = torch.concat([graph_repr, seq_repr], dim=1)
        fuse_repr = self.mol_fusion(fuse_repr)

        out = self.classifier(fuse_repr)
        return loss, out
