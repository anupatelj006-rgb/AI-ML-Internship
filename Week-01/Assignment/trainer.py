"""Training and validation loop."""
import torch
import torch.nn as nn
import config
from network import build_model


def _average_loss(model, loader, loss_fn):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            total += loss_fn(model(xb), yb).item() * len(xb)
            count += len(xb)
    return total / count


def train(train_loader, val_loader, lr, epochs=config.EPOCHS):
    model = build_model()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        running, count = 0.0, 0
        for xb, yb in train_loader:
            prediction = model(xb)               # forward pass
            loss = loss_fn(prediction, yb)       # measure error
            optimizer.zero_grad()                # reset gradients
            loss.backward()                      # backpropagation
            optimizer.step()                     # update weights and biases
            running += loss.item() * len(xb)
            count += len(xb)

        history["train_loss"].append(running / count)
        history["val_loss"].append(_average_loss(model, val_loader, loss_fn))

    return model, history