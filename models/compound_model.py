import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.nn as gnn
import torch_scatter
from data import MultiGraph

from models.compound_encoder import AtomEmbedding
from models.compound_encoder import BondEmbedding
from models.compound_encoder import BondFloatRBF
from models.compound_encoder import BondAngleFloatRBF
from models.gnn_block import GNNBlock

default_config = {
    "embed_dim": 64,
    "drop_out": 0.1,
    "layer_num": 8,
    "readout": "mean",
    "atom_names": [
        "atomic_num",
        "formal_charge",
        "degree",
        "chiral_tag",
        "total_numHs",
        "is_aromatic",
        "hybridization",
    ],
    "bond_names": ["bond_dir", "bond_type", "is_in_ring"],
    "bond_float_names": ["bond_length"],
    "bond_angle_float_names": ["bond_angle"],
}


class GNNModelBlock(nn.Module):
    def __init__(self, embed_dim, dropout, last_act):
        super().__init__()
        self.embed_dim = embed_dim
        self.last_act = last_act
        self.gnn = GNNBlock(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.graph_norm = gnn.GraphSizeNorm()
        self.dropout = nn.Dropout(p=dropout)
        if last_act:
            self.act = nn.ReLU()

    def forward(self, node_feat, edge_index, edge_attr):
        out = self.gnn(node_feat, edge_index, edge_attr)
        out = self.norm(out)
        out = self.graph_norm(out)
        out = self.dropout(out)
        if self.last_act:
            self.act(out)

        out = out + node_feat
        return out


class GNNModel(nn.Module):
    def __init__(self, model_config):
        super(GNNModel, self).__init__()

        self.embed_dim = model_config.get("embed_dim", 32)
        self.dropout_rate = model_config.get("dropout", 0.2)
        self.layer_num = model_config.get("layer_num", 8)
        self.readout_mode = model_config.get("readout", "mean")

        if self.readout_mode == "mean":
            self.readout = torch_scatter.scatter_mean
        elif self.readout_mode == "max":
            self.readout = torch_scatter.scatter_max
        elif self.readout_mode == "add":
            self.readout = torch_scatter.scatter_add
        else:
            raise ValueError("Unsupported Readout!")

        self.atom_names = model_config["atom_names"]
        self.bond_names = model_config["bond_names"]
        self.bond_float_names = model_config["bond_float_names"]
        self.bond_angle_float_names = model_config["bond_angle_float_names"]

        # Here we first init the atom embedding
        self.init_atom_embedding = AtomEmbedding(self.atom_names, self.embed_dim)

        # then, we init the attr bond embedding -> normal bond attr
        # including bond_dir, bond_type, is_in_ring

        self.init_bond_embedding = BondEmbedding(self.bond_names, self.embed_dim)

        # and the float bond embedding -> bond length
        self.init_bond_float_rbf = BondFloatRBF(self.bond_float_names, self.embed_dim)

        # separate attr
        self.bond_embedding_list = nn.ModuleList()
        self.bond_float_rbf_list = nn.ModuleList()
        self.bond_angle_float_rbf_list = nn.ModuleList()

        # here we mix the feat together
        self.atom_bond_block_list = nn.ModuleList()
        self.bond_angle_block_list = nn.ModuleList()

        for layer_id in range(self.layer_num):
            self.bond_embedding_list.append(
                BondEmbedding(self.bond_names, self.embed_dim)
            )

            self.bond_float_rbf_list.append(
                BondFloatRBF(self.bond_float_names, self.embed_dim)
            )

            self.bond_angle_float_rbf_list.append(
                BondAngleFloatRBF(self.bond_angle_float_names, self.embed_dim)
            )

            # add the GNNModelBlock network
            self.atom_bond_block_list.append(
                GNNModelBlock(
                    self.embed_dim,
                    dropout=self.dropout_rate,
                    last_act=(layer_id != self.layer_num - 1),
                )
            )

            self.bond_angle_block_list.append(
                GNNModelBlock(
                    self.embed_dim,
                    dropout=self.dropout_rate,
                    last_act=(layer_id != self.layer_num - 1),
                )
            )

        print("embed_dim:%s" % self.embed_dim)
        print("dropout_rate:%s" % self.dropout_rate)
        print("layer_num:%s" % self.layer_num)
        print("readout:%s" % self.readout_mode)
        print("atom_names:%s" % str(self.atom_names))
        print("bond_names:%s" % str(self.bond_names))
        print("bond_float_names:%s" % str(self.bond_float_names))
        print("bond_angle_float_names:%s" % str(self.bond_angle_float_names))

    @property
    def node_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    @property
    def graph_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    def forward(self, multi_graph: MultiGraph):
        """
        Build the network.
        """

        # init node hidden rep
        node_arr = multi_graph.x_a

        node_hidden = self.init_atom_embedding(node_arr)
        node_hidden = node_hidden.squeeze(1)
        # print("init node_hidden dim :{}".format(node_hidden.shape))

        # init bond hidden rep
        edge_int_arr = multi_graph.edge_attr_a[:, :-1]
        edge_int_attr = self.init_bond_embedding(edge_int_arr)
        edge_int_attr = edge_int_attr.squeeze(1)
        # print("init edge_int_attr dim :{}".format(edge_int_attr.shape))

        edge_float_arr = multi_graph.edge_attr_a[:, -1]
        edge_float_attr = self.init_bond_float_rbf(edge_float_arr)
        # print("init edge_float_attr dim :{}".format(edge_float_attr.shape))

        edge_hidden = edge_int_attr + edge_float_attr
        # print("init edge_hidden dim :{}".format(edge_hidden.shape))

        # init sep rep
        node_hidden_list = [node_hidden]
        edge_hidden_list = [edge_hidden]

        for layer_id in range(self.layer_num):
            # in the atom bond graph
            node_hidden = self.atom_bond_block_list[layer_id](
                node_hidden_list[layer_id],
                multi_graph.edge_index_a,
                edge_hidden_list[layer_id],
            )

            # print("{} -> node_hidden dim :{}".format(layer_id, node_hidden.shape))
            # cur_int_edge_hidden = self.bond_embedding_list[layer_id](atom_bond_graph["edge_feat"]).squeeze_(1)
            cur_int_edge_hidden = self.bond_embedding_list[layer_id](
                edge_int_arr
            ).squeeze_(1)

            # print("{} -> cur_int_edge_hidden dim :{}".format(layer_id, cur_int_edge_hidden.shape))
            # cur_float_edge_hidden = self.bond_float_rbf_list[layer_id]( atom_bond_graph["edge_feat"])
            cur_float_edge_hidden = self.bond_float_rbf_list[layer_id](edge_float_arr)

            # print("{} -> cur_float_edge_hidden dim :{}".format(layer_id, cur_float_edge_hidden.shape))
            # edge hidden is consisted of two parts the int and the float type features
            cur_edge_hidden = cur_int_edge_hidden + cur_float_edge_hidden

            # print("{} -> cur_edge_hidden dim :{}".format(layer_id, cur_edge_hidden.shape))
            # this is the angle rep
            # cur_angle_hidden = self.bond_angle_float_rbf_list[layer_id](bond_angle_graph["edge_feat"])
            cur_angle_hidden = self.bond_angle_float_rbf_list[layer_id](
                multi_graph.edge_attr_b
            )

            # print("{} -> cur_angle_hidden dim :{}".format(layer_id, cur_angle_hidden.shape))
            # in the bond angle graph -> here the edge as node, bond_angle_graph["edges"]

            edge_hidden = self.bond_angle_block_list[layer_id](
                cur_edge_hidden, multi_graph.edge_index_b, cur_angle_hidden
            )  # here the angle as the edge

            # print("{} -> edge_hidden dim :{}".format(layer_id, edge_hidden.shape))
            node_hidden_list.append(node_hidden)
            edge_hidden_list.append(edge_hidden)

        node_repr = node_hidden_list[-1]
        edge_repr = edge_hidden_list[-1]
        graph_repr = self.readout(node_repr, multi_graph.x_a_batch, dim=0)

        # print(graph_repr.shape)
        return graph_repr
