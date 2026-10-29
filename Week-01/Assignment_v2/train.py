import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

TRUE_W = torch.tensor([3.0, -2.0])   # the secret rule the model must discover
TRUE_B = 5.0


def make_data(n=1000, noise=0.5):
    """Synthetic data: y = 3*x1 - 2*x2 + 5 + noise."""
    X = torch.rand(n, 2) * 2 - 1
    y = (X @ TRUE_W + TRUE_B + noise * torch.randn(n)).unsqueeze(1)
    cut = int(n * 0.8)
    return (X[:cut], y[:cut]), (X[cut:], y[cut:])     # training, validation


def plot_data(train):
    X, y = train
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i in range(2):
        ax[i].scatter(X[:, i], y, s=8, alpha=0.5)
        ax[i].set_xlabel(f"Input x{i+1}"); ax[i].set_ylabel("Target y")
    plt.suptitle("Synthetic regression data")
    plt.savefig(os.path.join(OUT, "01_dataset.png"), dpi=150, bbox_inches="tight")
    plt.close()


def train_model(train, val, lr, epochs=30, batch_size=32):
    X_tr, y_tr = train
    X_va, y_va = val
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    model = nn.Linear(2, 1)                  # one neuron: 2 weights + 1 bias
    loss_fn = nn.MSELoss()                   # average squared mistake
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    hist = {"train": [], "val": []}

    for ep in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:                # one batch at a time
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()                  # find how each weight caused the error
            opt.step()                       # nudge the weights
            total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va), y_va).item()
        hist["train"].append(total / len(X_tr))
        hist["val"].append(val_loss)
    return model, hist


def plot_loss(hist, lr):
    plt.figure(figsize=(6, 4))
    plt.plot(hist["train"], label="Training loss")
    plt.plot(hist["val"], label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss (MSE)")
    plt.title(f"Learning rate = {lr}"); plt.legend()
    plt.savefig(os.path.join(OUT, f"02_loss_lr_{lr}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def compare(histories):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for lr, h in histories.items():
        ax[0].plot(h["train"], label=f"lr = {lr}")
        ax[1].plot(h["val"], label=f"lr = {lr}")
    ax[0].set_title("Training loss"); ax[1].set_title("Validation loss")
    for a in ax:
        a.set_xlabel("Epoch"); a.set_ylabel("Loss (MSE)"); a.set_yscale("log"); a.legend()
    plt.savefig(os.path.join(OUT, "03_lr_side_by_side.png"), dpi=150, bbox_inches="tight")
    plt.close()


def save_results(results):
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)