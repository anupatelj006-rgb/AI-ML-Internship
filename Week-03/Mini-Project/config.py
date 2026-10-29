import os
import numpy as np
import torch

SEED = 42
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")
MODEL_DIR = os.path.join(BASE, "saved_model")
DATA = os.path.join(BASE, "data")
os.makedirs(OUT, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "fashion_cnn.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Bigger categories: which of the 10 classes belong together
GROUPS = {"Tops": [0, 2, 4, 6], "Dress": [3], "Trousers": [1],
          "Footwear": [5, 7, 9], "Bags": [8]}
GROUP_NAMES = list(GROUPS)
CLASS_TO_GROUP = np.zeros(10, dtype=int)
for gi, members in enumerate(GROUPS.values()):
    for c in members:
        CLASS_TO_GROUP[c] = gi

MEAN, STD = 0.2860, 0.3530     # average and spread of the pixels
EPOCHS = 8                     # use 4 for a quick test run


def set_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)