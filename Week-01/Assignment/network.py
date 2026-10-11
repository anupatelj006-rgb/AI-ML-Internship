"""A small feed-forward network: 1 input -> hidden layer (ReLU) -> 1 output."""
import torch
import torch.nn as nn
import config


class RegressionNet(nn.Module):
    def __init__(self, hidden=config.HIDDEN_UNITS):
        super().__init__()
        self.hidden_layer = nn.Linear(1, hidden)   # weights + biases of hidden neurons
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.activation(x)
        return self.output_layer(x)


def build_model():
    torch.manual_seed(config.MODEL_SEED)   # identical start for every run
    return RegressionNet()