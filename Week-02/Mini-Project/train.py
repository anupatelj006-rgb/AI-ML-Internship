import os
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from config import OUT
from model import build_model, DESIGNS


def accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(1) == y).float().mean().item()


def train_model(model, train, val, epochs=10, lr=1e-3, batch=128, verbose=True):
    (x_train, y_train), (x_val, y_val) = train, val
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(x_train))
        total = 0
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(x_train[b]), y_train[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), y_val).item()
        hist["train_loss"].append(total / len(x_train))
        hist["val_loss"].append(val_loss)
        hist["train_acc"].append(accuracy(model, x_train, y_train))
        hist["val_acc"].append(accuracy(model, x_val, y_val))
        if verbose:
            print(f"Epoch {ep+1}/{epochs} | train acc {hist['train_acc'][-1]:.4f} "
                  f"| val acc {hist['val_acc'][-1]:.4f}")
    return hist


def compare_designs(train, val):
    """Tries every design for 5 epochs and ranks them on validation data."""
    rows = []
    for name, cfg in DESIGNS.items():
        m = build_model(**cfg)
        h = train_model(m, train, val, epochs=5, verbose=False)
        rows.append({"Design": name,
                     "Train acc": h["train_acc"][-1],
                     "Val acc": h["val_acc"][-1],
                     "Gap (train - val)": h["train_acc"][-1] - h["val_acc"][-1]})
        print("Compared:", name)

    table = pd.DataFrame(rows).sort_values("Val acc", ascending=False).reset_index(drop=True)
    table.to_csv(os.path.join(OUT, "settings_comparison.csv"), index=False)

    plt.figure(figsize=(7, 3.5))
    plt.barh(table["Design"][::-1], table["Val acc"][::-1], color="#4c9f70")
    plt.xlim(table["Val acc"].min() - 0.01, 1.0)
    plt.xlabel("Validation accuracy")
    plt.title("Settings comparison")
    plt.savefig(os.path.join(OUT, "settings_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    return table


def plot_curves(hist):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist["train_acc"], label="Training")
    plt.plot(hist["val_acc"], label="Validation")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(hist["train_loss"], label="Training")
    plt.plot(hist["val_loss"], label="Validation")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()
    plt.savefig(os.path.join(OUT, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()