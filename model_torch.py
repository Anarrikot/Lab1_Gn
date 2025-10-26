# model_torch.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.SiLU()  # swish-like
        self.drop = nn.Dropout(dropout) if dropout>0 else nn.Identity()

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class TorchRegressor(nn.Module):
    """
    "EfficientNet-like" lite architecture for tabular regression:
    stack of MLPBlocks -> final linear output.
    """
    def __init__(self, input_dim, hidden=[128,64], dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(MLPBlock(prev, h, dropout))
            prev = h
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, x):
        x = self.net(x)
        out = self.head(x)
        return out
