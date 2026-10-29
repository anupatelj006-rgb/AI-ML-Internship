import os
import numpy as np
import torch

SEED = 42
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")
DATA = os.path.join(BASE, "data")
os.makedirs(OUT, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# average and spread of Fashion-MNIST pixels (used for normalization)
MEAN, STD = 0.2860, 0.3530

EPOCHS = 8


def set_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)