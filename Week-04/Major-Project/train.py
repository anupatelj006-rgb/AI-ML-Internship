import copy
import time

import torch
import torch.nn as nn

import config
from models import build_model, count_params


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, correct / n


def train_model(name, loaders, device, epochs=config.EPOCHS):
    model = build_model(name, pretrained=(name == "transfer")).to(device)
    criterion = nn.CrossEntropyLoss()
    warmup = config.WARMUP_EPOCHS if name == "transfer" else 0
    scheduler = None

    if name == "baseline":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    elif name == "improved":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    else:  # transfer: phase 1 trains only the new head
        model.set_backbone_trainable(False)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=1e-3, weight_decay=1e-4)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_state, best_epoch, wait = 0.0, None, 0, 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        if warmup and epoch == warmup + 1:  # phase 2: fine-tune everything with a small LR
            model.set_backbone_trainable(True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
            wait = 0
            print(f"[{name}] backbone unfrozen, fine-tuning")

        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, loaders["val"], criterion, device)
        if scheduler:
            scheduler.step(va_loss)
        for k, v in zip(history, (tr_loss, va_loss, tr_acc, va_acc)):
            history[k].append(v)
        print(f"[{name}] epoch {epoch:02d}/{epochs} "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | val loss {va_loss:.3f} acc {va_acc:.3f}")

        if va_acc > best_acc:
            best_acc, best_epoch, wait = va_acc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if epoch > warmup and wait >= config.PATIENCE:
                print(f"[{name}] early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    info = {
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "epochs_run": len(history["val_acc"]),
        "train_seconds": round(time.time() - start, 1),
        "params": count_params(model),
    }
    if name == "transfer":
        info["pretrained_loaded"] = bool(model.pretrained_loaded)

    config.MODEL_DIR.mkdir(exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "classes": config.CLASSES,
        "img_size": config.IMG_SIZE,
        "history": history,
        "info": info,
        "temperature": 1.0,
    }, config.MODEL_DIR / f"{name}.pt")
    return model, history, info