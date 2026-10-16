import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from config import OUT
from train import accuracy


def confidence_analysis(pred, y_true, conf):
    plt.figure(figsize=(7, 4))
    plt.hist(conf[pred == y_true], bins=20, alpha=0.7, label="Correct", color="green")
    plt.hist(conf[pred != y_true], bins=20, alpha=0.7, label="Wrong", color="red")
    plt.yscale("log")
    plt.xlabel("Confidence"); plt.ylabel("Number of images (log scale)")
    plt.title("Confidence: correct vs wrong"); plt.legend()
    plt.savefig(os.path.join(OUT, "confidence_hist.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nAverage confidence when correct: {conf[pred == y_true].mean():.3f}")
    print(f"Average confidence when wrong:   {conf[pred != y_true].mean():.3f}")


def trust_threshold(pred, y_true, conf):
    rows = []
    for t in [0.0, 0.5, 0.7, 0.9, 0.95, 0.99]:
        keep = conf >= t
        rows.append({"Threshold": t,
                     "Images answered (%)": keep.mean() * 100,
                     "Accuracy on those (%)": (pred[keep] == y_true[keep]).mean() * 100})
    table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(table["Threshold"], table["Accuracy on those (%)"], "o-", label="Accuracy", color="green")
    ax.plot(table["Threshold"], table["Images answered (%)"], "s--", label="Images answered", color="orange")
    ax.set_xlabel("Confidence threshold"); ax.set_ylabel("%"); ax.legend()
    ax.set_title("Accuracy vs how many images the model answers")
    plt.savefig(os.path.join(OUT, "confidence_tradeoff.png"), dpi=150, bbox_inches="tight")
    plt.close()
    table.to_csv(os.path.join(OUT, "trust_threshold.csv"), index=False)
    print("\n", table.round(2).to_string(index=False))


def neuron_weight_maps(model):
    first = [m for m in model if isinstance(m, nn.Linear)][0]
    W = first.weight.detach().numpy()
    plt.figure(figsize=(10, 6))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(W[i].reshape(28, 28), cmap="RdBu"); plt.axis("off")
        plt.title(f"Neuron {i}", fontsize=8)
    plt.suptitle("First-layer neurons: what each one looks for")
    plt.savefig(os.path.join(OUT, "neuron_weight_maps.png"), dpi=150, bbox_inches="tight")
    plt.close()


def stress_test(model, x_test, y_test):
    noise, rot, shift = [0.0, 0.2, 0.4, 0.6], [0, 10, 20, 30], [0, 2, 4, 6]
    x4 = x_test.unsqueeze(1)

    r_noise = [accuracy(model, (x_test + s * torch.randn_like(x_test)).clamp(0, 1), y_test) for s in noise]
    r_rot = [accuracy(model, TF.rotate(x4, a).squeeze(1), y_test) for a in rot]
    r_shift = [accuracy(model, torch.roll(x_test, s, dims=2), y_test) for s in shift]

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    for a, xs, ys, title in zip(ax, [noise, rot, shift], [r_noise, r_rot, r_shift],
                                ["Added noise", "Rotation (degrees)", "Shift (pixels)"]):
        a.plot(xs, ys, "o-"); a.set_title(title)
        a.set_ylabel("Test accuracy"); a.set_ylim(0, 1)
    plt.savefig(os.path.join(OUT, "stress_test.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("\nStress test -> noise:", np.round(r_noise, 3),
          "| rotation:", np.round(r_rot, 3), "| shift:", np.round(r_shift, 3))