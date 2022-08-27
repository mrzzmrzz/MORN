"""
Basic Encoder for compound atom/bond features.
"""

import numpy as np
import torch
import torch.nn as nn
from models.basic_block import RBF
from utils.compound_tools import CompoundKit


def init_weight(m: nn.Module):
    class_name = m.__class__.__name__

    if class_name == "Embedding":
        nn.init.xavier_normal_(m.weight.data)

    elif class_name == "Linear":
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class AtomEmbedding(nn.Module):
    """
    Atom Encoder
    """

    def __init__(self, atom_names, embed_dim):
        super(AtomEmbedding, self).__init__()
        self.atom_names = atom_names

        self.embed_list = nn.ModuleList()
        for name in self.atom_names:
            embed = nn.Embedding(CompoundKit.get_atom_feature_size(name) + 5, embed_dim)
            self.embed_list.append(embed)

        self.apply(init_weight)

    def forward(self, node_features):
        """
        Args:
            node_features(dict of tensor): node features.
        """

        out_embed = 0
        for i, name in enumerate(self.atom_names):
            out_embed += self.embed_list[i](node_features[:, i])
        return out_embed


class AtomFloatEmbedding(nn.Module):
    """
    Atom Float Encoder
    """

    def __init__(self, atom_float_names, embed_dim, rbf_params=None):
        super(AtomFloatEmbedding, self).__init__()
        self.atom_float_names = atom_float_names

        if rbf_params is None:
            self.rbf_params = {
                "van_der_waals_radis": (np.arange(1, 3, 0.2), 10.0),  # (centers, gamma)
                "partial_charge": (np.arange(-1, 4, 0.25), 10.0),  # (centers, gamma)
                "mass": (np.arange(0, 2, 0.1), 10.0),  # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        for name in self.atom_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim)
            self.linear_list.append(linear)

        self.apply(init_weight)

    def forward(self, feats):
        """
        Args:
            feats(dict of tensor): node float features.
        """
        out_embed = 0
        for i, name in enumerate(self.atom_float_names):
            x = feats[
                :,
            ]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        return out_embed


class BondEmbedding(nn.Module):
    """
    Bond Encoder
    """

    def __init__(self, bond_names, embed_dim):
        super(BondEmbedding, self).__init__()
        self.bond_names = bond_names

        self.embed_list = nn.ModuleList()
        for name in self.bond_names:
            embed = nn.Embedding(CompoundKit.get_bond_feature_size(name) + 5, embed_dim)
            self.embed_list.append(embed)
        self.apply(init_weight)

    def forward(self, edge_features):
        """
        Args:
            edge_features(dict of tensor): edge features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_names):
            out_embed += self.embed_list[i](edge_features[:, i].type(torch.int))
        return out_embed


class BondFloatRBF(nn.Module):
    """
    Bond Float Encoder using Radial Basis Functions
    """

    def __init__(self, bond_float_names, embed_dim, rbf_params=None):
        super(BondFloatRBF, self).__init__()
        self.bond_float_names = bond_float_names

        if rbf_params is None:

            self.rbf_params = {
                "bond_length": (np.arange(0, 2, 0.1), 10.0),  # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        for name in self.bond_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim)
            self.linear_list.append(linear)
        self.apply(init_weight)

    def forward(self, bond_float_features):
        """
        Args:
            bond_float_features(tensor): bond float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_float_names):
            x = bond_float_features
            rbf_x = self.rbf_list[i](x.type(torch.float))
            out_embed += self.linear_list[i](rbf_x)
        return out_embed


class BondAngleFloatRBF(nn.Module):
    """
    Bond Angle Float Encoder using Radial Basis Functions
    """

    def __init__(self, bond_angle_float_names, embed_dim, rbf_params=None):
        super(BondAngleFloatRBF, self).__init__()
        self.bond_angle_float_names = bond_angle_float_names

        if rbf_params is None:
            self.rbf_params = {
                "bond_angle": (np.arange(0, np.pi, 0.1), 10.0),  # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        for name in self.bond_angle_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim)
            self.linear_list.append(linear)
        self.apply(init_weight)

    def forward(self, bond_angle_float_features):
        """
        Args:
            bond_angle_float_features (tensor): bond angle float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_angle_float_names):
            x = bond_angle_float_features
            rbf_x = self.rbf_list[i](x.type(torch.float))
            out_embed += self.linear_list[i](rbf_x)
        return out_embed


if __name__ == "__main__":
    print("OK")
