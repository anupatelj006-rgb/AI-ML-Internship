import time
import torch
import torch.nn as nn
from config import DEVICE


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = loss_fn(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def fit(model, train_loader, val_loader, epochs=8, lr=1e-3, name="model"):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    start = time.time()

    for ep in range(epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, opt)
        va_loss, va_acc = run_epoch(model, val_loader, loss_fn)
        hist["train_loss"].append(tr_loss); hist["train_acc"].append(tr_acc)
        hist["val_loss"].append(va_loss);   hist["val_acc"].append(va_acc)
        print(f"[{name}] epoch {ep+1}/{epochs} | train acc {tr_acc:.4f} | val acc {va_acc:.4f}")

    hist["seconds"] = time.time() - start
    return hist