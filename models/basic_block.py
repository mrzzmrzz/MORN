import torch
import torch.nn as nn
from config import cuda_device_num

if cuda_device_num >= 0:
    device = torch.device("cuda:{}".format(cuda_device_num))
else:
    device = torch.device("cpu")


class RBF(nn.Module):
    """
    Radial Basis Function
    """

    def __init__(self, centers, gamma, dtype=torch.float32):
        super(RBF, self).__init__()
        self.centers = torch.tensor(centers, dtype=dtype, device=device)
        self.centers = self.centers.view(1, -1)
        self.gamma = torch.tensor(gamma, device=device)

    def forward(self, x):
        """
        Args:
            x(tensor): (-1, 1).
        Returns:
            y(tensor): (-1, n_centers)
        """
        x = x.view(-1, 1)
        return torch.exp(-self.gamma * torch.square(x - self.centers))


class MLP(nn.Module):
    """
    MLP
    """

    def __init__(self, config):
        super(MLP, self).__init__()

        self.layer_num = config["layer_num"]
        self.input_dim = config["input_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.output_dim = config["output_dim"]
        self.dropout_rate = config["dropout_rate"]

        layers = []
        for layer_id in range(self.layer_num):
            if layer_id == 0:
                layers.append(nn.Linear(self.input_dim, self.hidden_dim))
                layers.append(nn.Dropout(self.dropout_rate))
                layers.append(nn.LeakyReLU())
            elif layer_id < self.layer_num - 1:
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(nn.Dropout(self.dropout_rate))
                layers.append(nn.LeakyReLU())
            else:
                layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x(tensor): (-1, dim).
        """
        return self.mlp(x)
