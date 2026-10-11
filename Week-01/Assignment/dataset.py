"""Synthetic regression data: y = slope * x + intercept + noise."""
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import config


def make_data():
    gen = torch.Generator().manual_seed(config.DATA_SEED)
    x = torch.rand(config.N_SAMPLES, 1, generator=gen)
    noise = config.NOISE_STD * torch.randn(config.N_SAMPLES, 1, generator=gen)
    y = config.TRUE_SLOPE * x + config.TRUE_INTERCEPT + noise
    return x, y


def make_loaders(x, y):
    full = TensorDataset(x, y)
    val_size = len(full) - config.TRAIN_SIZE
    train_set, val_set = random_split(
        full, [config.TRAIN_SIZE, val_size],
        generator=torch.Generator().manual_seed(config.DATA_SEED),
    )
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE)
    return train_loader, val_loader