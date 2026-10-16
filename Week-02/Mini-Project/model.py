import torch.nn as nn


def build_model(hidden=(128, 64), activation=nn.ReLU, dropout=0.2):
    layers, prev = [nn.Flatten()], 784        # 28x28 picture -> 784 numbers
    for h in hidden:
        layers += [nn.Linear(prev, h), activation(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, 10))         # 10 outputs, one per digit
    return nn.Sequential(*layers)


DESIGNS = {
    "Small (64)":               dict(hidden=(64,),      activation=nn.ReLU, dropout=0.0),
    "Medium (128-64, dropout)": dict(hidden=(128, 64),  activation=nn.ReLU, dropout=0.2),
    "Wide (256-128, dropout)":  dict(hidden=(256, 128), activation=nn.ReLU, dropout=0.3),
    "Tanh (128-64)":            dict(hidden=(128, 64),  activation=nn.Tanh, dropout=0.2),
}