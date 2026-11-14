import os
import numpy as np
import torch

SEED = 42
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)


def _find_data():
    """Reuse the Fashion-MNIST download from Week 3 if it is already on this computer."""
    candidates = [
        os.path.join(BASE, "data"),
        os.path.join(BASE, "..", "..", "Week-03", "Mini-Project", "data"),
        os.path.join(BASE, "..", "..", "Week-03", "Assignment", "data"),
    ]
    for p in candidates:
        if os.path.isdir(os.path.join(p, "FashionMNIST", "raw")):
            return os.path.abspath(p)
    return os.path.join(BASE, "data")


DATA = _find_data()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

MEAN, STD = 0.2860, 0.3530     # average and spread of the pixels

# The model configurations to compare. For a quick test, set both "epochs" to 1.
CONFIGS = [
    {"name": "Config A: Small CNN", "model": "small", "dropout": 0.3,
     "lr": 1e-3, "weight_decay": 0.0, "epochs": 8, "scheduler": "none"},
    {"name": "Config B: Deep CNN (tuned)", "model": "deep", "dropout": 0.4,
     "lr": 1e-3, "weight_decay": 1e-4, "epochs": 10, "scheduler": "cosine"},
]


def set_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)