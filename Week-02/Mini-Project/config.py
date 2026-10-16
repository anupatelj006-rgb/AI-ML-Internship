import os
import numpy as np
import torch

SEED = 42
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")
DATA = os.path.join(BASE, "data")
os.makedirs(OUT, exist_ok=True)


def set_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)