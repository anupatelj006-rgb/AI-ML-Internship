"""All plots are saved as PNG files in the outputs folder."""
import os
import torch
import matplotlib.pyplot as plt
import config


def _save(name, show):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.OUTPUT_DIR, name), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_dataset(x, y, show=False):
    plt.figure(figsize=(6, 4))
    plt.scatter(x.numpy(), y.numpy(), s=6, alpha=0.5)
    plt.xlabel("x"); plt.ylabel("y")
    plt.title("Synthetic dataset")
    _save("01_dataset.png", show)


def plot_loss_curves(history, lr, show=False):
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="Training loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE loss")
    plt.title(f"Training and validation loss (lr = {lr})")
    plt.legend(); plt.grid(alpha=0.3)
    _save("02_loss_curves.png", show)


def plot_prediction(model, x, y, show=False):
    model.eval()
    grid = torch.linspace(0, 1, 100).unsqueeze(1)
    with torch.no_grad():
        pred = model(grid)
    plt.figure(figsize=(6, 4))
    plt.scatter(x.numpy(), y.numpy(), s=6, alpha=0.4, label="Data")
    plt.plot(grid.numpy(), pred.numpy(), "r-", linewidth=2, label="Model")
    plt.plot(grid.numpy(),
             config.TRUE_SLOPE * grid.numpy() + config.TRUE_INTERCEPT,
             "k--", label="True function")
    plt.xlabel("x"); plt.ylabel("y"); plt.legend()
    plt.title("Model prediction vs true function")
    _save("03_prediction.png", show)


def plot_lr_comparison(all_history, show=False):
    fig, axes = plt.subplots(1, len(all_history), figsize=(6 * len(all_history), 4),
                             sharey=True)
    for ax, (lr, h) in zip(axes, all_history.items()):
        ax.plot(h["train_loss"], label="Training loss")
        ax.plot(h["val_loss"], label="Validation loss")
        ax.set_title(f"Learning rate = {lr}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss")
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    _save("04_lr_side_by_side.png", show)

    plt.figure(figsize=(7, 4))
    for lr, h in all_history.items():
        plt.plot(h["val_loss"], label=f"lr = {lr}")
    plt.yscale("log")
    plt.xlabel("Epoch"); plt.ylabel("Validation MSE (log scale)")
    plt.title("Validation loss for different learning rates")
    plt.legend(); plt.grid(alpha=0.3, which="both")
    _save("05_lr_validation_log.png", show)