import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basic_block import MLP
from models.compound_model import GNNModel
from models.seq_block import SeqEncoder


class GNNPreModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.gnnModel = GNNModel(config["gnn_config"])
        self.classifier = MLP(config["classifier_config"])

    def forward(self, data):
        graph_repr = self.gnnModel(data)
        return self.classifier(graph_repr)


class SeqPreModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.seqModel = SeqEncoder(config["seq_config"])
        self.classifier = MLP(config["classifier_config"])

    def forward(self, data):
        seq_repr = self.seqModel(data.seq_feat, data.seq_len)
        return self.classifier(seq_repr)
