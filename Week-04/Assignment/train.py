import copy
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


def fit(model, train_loader, val_loader, cfg):
    """Trains with the settings in cfg and keeps the best version (by validation accuracy)."""
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = None
    if cfg["scheduler"] == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    loss_fn = nn.CrossEntropyLoss()

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_state = -1.0, None
    start = time.time()

    for ep in range(cfg["epochs"]):
        tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, opt)
        va_loss, va_acc = run_epoch(model, val_loader, loss_fn)
        if sched:
            sched.step()
        hist["train_loss"].append(tr_loss); hist["train_acc"].append(tr_acc)
        hist["val_loss"].append(va_loss);   hist["val_acc"].append(va_acc)
        if va_acc > best_acc:
            best_acc, best_state = va_acc, copy.deepcopy(model.state_dict())
            hist["best_epoch"] = ep + 1
        print(f"[{cfg['name']}] epoch {ep+1}/{cfg['epochs']} | train acc {tr_acc:.4f} | val acc {va_acc:.4f}")

    model.load_state_dict(best_state)
    hist["best_val_acc"] = best_acc
    hist["seconds"] = time.time() - start
    return hist