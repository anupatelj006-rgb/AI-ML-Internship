import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from config import OUT, DEVICE, CLASSES, MEAN, STD


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def predict(model, loader):
    model.eval()
    preds, trues, confs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            probs = torch.softmax(model(x.to(DEVICE)), dim=1)
            conf, pred = probs.max(1)
            preds.append(pred.cpu().numpy())
            trues.append(y.numpy())
            confs.append(conf.cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues), np.concatenate(confs)


def plot_class_distribution(dist):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bottom = np.zeros(len(dist))
    for split, color in [("Training", "#4c9f70"), ("Validation", "#e0a030"), ("Test", "#4c78a8")]:
        values = dist[split].values.astype(float)
        ax.bar(dist["Class"], values, bottom=bottom, label=split, color=color)
        bottom += values
    ax.set_ylabel("Number of images")
    ax.set_title("Class distribution by split")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.savefig(os.path.join(OUT, "01_class_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_curves(histories):
    n = len(histories)
    fig, ax = plt.subplots(n, 2, figsize=(12, 4.2 * n), squeeze=False)
    for row, (name, h) in enumerate(histories.items()):
        epochs = range(1, len(h["train_acc"]) + 1)
        ax[row][0].plot(epochs, h["train_acc"], label="Training")
        ax[row][0].plot(epochs, h["val_acc"], label="Validation")
        ax[row][0].set_title(f"{name}: accuracy")
        ax[row][1].plot(epochs, h["train_loss"], label="Training")
        ax[row][1].plot(epochs, h["val_loss"], label="Validation")
        ax[row][1].set_title(f"{name}: loss")
        for a in ax[row]:
            a.set_xlabel("Epoch")
            a.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "02_training_validation_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


def per_class_metrics(true, pred):
    p, r, f, s = precision_recall_fscore_support(true, pred, labels=list(range(10)), zero_division=0)
    table = pd.DataFrame({"Class": CLASSES, "Precision": p, "Recall": r, "F1": f, "Support": s})
    table.to_csv(os.path.join(OUT, "04_per_class_results.csv"), index=False)
    return table


def plot_per_class(table):
    t = table.sort_values("F1")
    plt.figure(figsize=(7, 5))
    plt.barh(t["Class"], t["F1"], color="#4c9f70")
    plt.xlim(0.5, 1.0)
    plt.xlabel("F1 score (test set)")
    plt.title("F1 score by clothing type")
    plt.savefig(os.path.join(OUT, "04_per_class_f1.png"), dpi=150, bbox_inches="tight")
    plt.close()


def confusion_plot(true, pred):
    cm = confusion_matrix(true, pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues",
                                                           colorbar=False, xticks_rotation=45)
    plt.title("Confusion matrix (test set)")
    plt.savefig(os.path.join(OUT, "05_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    return cm


def top_confusions(cm, k=5):
    errs = cm.copy()
    np.fill_diagonal(errs, 0)
    pairs = []
    for idx in np.argsort(errs.ravel())[::-1][:k]:
        t, p = divmod(int(idx), 10)
        pairs.append((CLASSES[t], CLASSES[p], int(errs[t, p])))
    return pairs


def failure_gallery(test_ds, pred, true, conf, n=10, cols=5):
    """The n wrong predictions the model was MOST sure about."""
    wrong = np.where(pred != true)[0]
    order = wrong[np.argsort(-conf[wrong])][:n]
    rows = int(np.ceil(len(order) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(2.6 * cols, 3.2 * rows))
    ax = np.array(ax).reshape(-1)
    for a in ax:
        a.axis("off")
    failures = []
    for a, i in zip(ax, order):
        img, _ = test_ds[int(i)]
        a.imshow(img.squeeze() * STD + MEAN, cmap="gray")
        a.set_title(f"True: {CLASSES[true[i]]}\nPred: {CLASSES[pred[i]]}\n{conf[i]*100:.0f}% sure",
                    color="red", fontsize=9)
        failures.append({"index": int(i), "true": CLASSES[true[i]],
                         "pred": CLASSES[pred[i]], "conf": float(conf[i])})
    plt.suptitle("Failure examples: confident wrong predictions", fontsize=13)
    plt.savefig(os.path.join(OUT, "06_failure_examples.png"), dpi=150, bbox_inches="tight")
    plt.close()
    return failures


def confidence_stats(pred, true, conf, thr=0.9):
    ok = pred == true
    keep = conf >= thr
    return {"conf_correct": float(conf[ok].mean()), "conf_wrong": float(conf[~ok].mean()),
            "thr": thr, "keep_rate": float(keep.mean()), "acc_keep": float(ok[keep].mean())}


def accuracy_ci(acc, n):
    """95% confidence interval for an accuracy measured on n images."""
    se = (acc * (1 - acc) / n) ** 0.5
    return acc - 1.96 * se, acc + 1.96 * se