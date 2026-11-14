"""Evaluation: metrics, calibration, significance test, error analysis, robustness."""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from scipy.stats import binomtest
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import config
from data_loader import get_eval_transform, load_image


@torch.no_grad()
def get_logits(model, loader, device):
    model.eval()
    out, ys = [], []
    for x, y in loader:
        out.append(model(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(out)


def fit_temperature(logits, labels):
    """Temperature scaling: learn one number T so softmax(logits / T) is better calibrated."""
    labels = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def save_temperature(name, temperature):
    path = config.MODEL_DIR / f"{name}.pt"
    ckpt = torch.load(path, map_location="cpu")
    ckpt["temperature"] = float(temperature)
    torch.save(ckpt, path)


def ece(probs, y, bins=10):
    """Expected calibration error: average gap between confidence and accuracy."""
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(total)


def reliability_points(probs, y, bins=10):
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        xs.append((lo + hi) / 2)
        ys.append(acc[m].mean() if m.any() else np.nan)
    return xs, ys


def plot_reliability(raw, cal, y, title, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    for p, label in ((raw, "before scaling"), (cal, "after scaling")):
        xs, ys = reliability_points(p, y)
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("Model score (confidence)")
    ax.set_ylabel("Actual accuracy")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def wilson_ci(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def mcnemar_p(y, pred_a, pred_b):
    """Exact McNemar test: do the two models disagree in one direction more than chance?"""
    a_ok, b_ok = pred_a == y, pred_b == y
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    if b + c == 0:
        return 1.0
    return float(binomtest(b, b + c, 0.5).pvalue)


def sorted_wrong(probs, y):
    """Indices of wrong predictions, most confident mistakes first."""
    wrong = np.where(probs.argmax(1) != y)[0]
    return wrong[np.argsort(-probs.max(1)[wrong])]


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cm, cmap="Blues")
    n = len(config.CLASSES)
    ax.set_xticks(range(n)); ax.set_xticklabels(config.CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(config.CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=10)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_errors(samples, y_true, y_pred, probs, wrong_idx, title, path, max_images=12):
    if len(wrong_idx) == 0:
        return None
    chosen = wrong_idx[:max_images]
    cols = 4
    rows = math.ceil(len(chosen) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, chosen):
        ax.imshow(load_image(samples[i][0]))
        ax.set_title(f"true: {config.CLASSES[y_true[i]]}\npred: {config.CLASSES[y_pred[i]]} "
                     f"({probs[i, y_pred[i]]:.0%})", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def evaluate_probs(name, probs, raw_probs, y, samples):
    """probs: calibrated probabilities (N, C). raw_probs: before temperature scaling."""
    pred = probs.argmax(1)
    n = len(y)
    correct = int((pred == y).sum())
    lo, hi = wilson_ci(correct, n)
    labels = list(range(len(config.CLASSES)))
    cm = confusion_matrix(y, pred, labels=labels)
    report = classification_report(y, pred, labels=labels, target_names=config.CLASSES,
                                   output_dict=True, zero_division=0)
    conf = probs.max(1)
    wrong = sorted_wrong(probs, y)
    off = [(config.CLASSES[i], config.CLASSES[j], int(cm[i, j]))
           for i in labels for j in labels if i != j and cm[i, j] > 0]
    off.sort(key=lambda t: -t[2])

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    title = config.MODEL_TITLES[name]
    cm_path = config.OUTPUT_DIR / f"confusion_{name}.png"
    err_path = config.OUTPUT_DIR / f"errors_{name}.png"
    rel_path = config.OUTPUT_DIR / f"reliability_{name}.png"
    plot_confusion(cm, f"{title} - test confusion matrix", cm_path)
    plot_errors(samples, y, pred, probs, wrong, f"{title} - most confident mistakes", err_path)
    plot_reliability(raw_probs, probs, y, f"{title} - reliability", rel_path)

    return {
        "n_test": n,
        "correct": correct,
        "accuracy": correct / n,
        "ci_low": lo,
        "ci_high": hi,
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "per_class": {c: {k: float(v) for k, v in report[c].items()} for c in config.CLASSES},
        "confusion_matrix": cm.tolist(),
        "n_wrong": int(len(wrong)),
        "top_confusions": [list(t) for t in off[:5]],
        "mean_conf_correct": float(conf[pred == y].mean()),
        "mean_conf_wrong": float(conf[wrong].mean()) if len(wrong) else 0.0,
        "wrong_conf_ge_80": int((conf[wrong] >= 0.8).sum()),
        "wrong_conf_ge_60": int((conf[wrong] >= 0.6).sum()),
        "ece_raw": ece(raw_probs, y),
        "ece_cal": ece(probs, y),
        "mcnemar_vs_baseline": None,
        "confusion_plot": cm_path.name,
        "error_plot": err_path.name if len(wrong) else None,
        "reliability_plot": rel_path.name,
    }


def _add_noise(im):
    a = np.asarray(im).astype(np.float32)
    a = a + np.random.normal(0, 25, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def robustness(model, samples, device):
    """Accuracy of one model when test images are degraded (lighting, blur, rotation, noise)."""
    model.eval()
    transform = get_eval_transform()
    tests = {
        "original": lambda im: im,
        "darker (brightness x0.5)": lambda im: ImageEnhance.Brightness(im).enhance(0.5),
        "brighter (brightness x1.6)": lambda im: ImageEnhance.Brightness(im).enhance(1.6),
        "blurred (radius 2)": lambda im: im.filter(ImageFilter.GaussianBlur(2)),
        "rotated 30 degrees": lambda im: im.rotate(30, resample=Image.BILINEAR),
        "low resolution (32 px)": lambda im: im.resize((32, 32)).resize(im.size),
        "noisy (sigma 25)": _add_noise,
    }
    out = {}
    for tname, fn in tests.items():
        correct = 0
        for s in range(0, len(samples), 64):
            batch = samples[s:s + 64]
            x = torch.stack([transform(fn(load_image(p))) for p, _ in batch]).to(device)
            y = torch.tensor([label for _, label in batch])
            with torch.no_grad():
                pred = model(x).argmax(1).cpu()
            correct += int((pred == y).sum())
        out[tname] = correct / len(samples)
    return out


def plot_histories(histories, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in histories.items():
        label = config.MODEL_TITLES[name].split(" (")[0]
        axes[0].plot(h["val_loss"], label=f"{label} val")
        axes[0].plot(h["train_loss"], "--", label=f"{label} train")
        axes[1].plot(h["val_acc"], label=f"{label} val")
        axes[1].plot(h["train_acc"], "--", label=f"{label} train")
    axes[0].set_title("Loss")
    axes[1].set_title("Accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)