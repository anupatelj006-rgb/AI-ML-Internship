import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from config import OUT, DEVICE, CLASSES, CLASS_TO_GROUP, MEAN, STD, SEED


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


def category_accuracy(true, pred):
    """Right bigger category (Tops, Footwear...) even if the exact item is wrong."""
    return float((CLASS_TO_GROUP[true] == CLASS_TO_GROUP[pred]).mean())


def plot_curves(histories):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for name, h in histories.items():
        ax[0].plot(h["train_acc"], label=name)
        ax[1].plot(h["val_acc"], label=name)
        ax[2].plot(h["val_loss"], label=name)
    ax[0].set_title("Training accuracy")
    ax[1].set_title("Validation accuracy")
    ax[2].set_title("Validation loss")
    for a in ax:
        a.set_xlabel("Epoch")
    ax[0].legend(fontsize=8)
    plt.savefig(os.path.join(OUT, "02_training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


def per_class_accuracy(true, pred, name):
    rows = [{"Class": CLASSES[c], "Accuracy": float((pred[true == c] == c).mean())} for c in range(10)]
    table = pd.DataFrame(rows).sort_values("Accuracy").reset_index(drop=True)
    table.to_csv(os.path.join(OUT, "06_per_class_accuracy.csv"), index=False)
    plt.figure(figsize=(7, 5))
    plt.barh(table["Class"], table["Accuracy"], color="#4c9f70")
    plt.xlim(0.5, 1.0)
    plt.xlabel("Test accuracy")
    plt.title(f"Accuracy by clothing type ({name})")
    plt.savefig(os.path.join(OUT, "06_per_class_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close()
    return table


def confusion(true, pred, name):
    cm = confusion_matrix(true, pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues",
                                                           colorbar=False, xticks_rotation=45)
    plt.title(f"Confusion matrix ({name})")
    plt.savefig(os.path.join(OUT, "07_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()


def gallery(test_ds, idxs, pred, true, conf, filename, title, cols=6):
    rows = int(np.ceil(len(idxs) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.9 * rows))
    ax = np.array(ax).reshape(-1)
    for a in ax:
        a.axis("off")
    for a, i in zip(ax, idxs):
        img, _ = test_ds[int(i)]
        a.imshow(img.squeeze() * STD + MEAN, cmap="gray")
        ok = pred[i] == true[i]
        a.set_title(f"Pred: {CLASSES[pred[i]]}\n(true: {CLASSES[true[i]]})\n{conf[i]*100:.0f}% sure",
                    color="green" if ok else "red", fontsize=8)
    plt.suptitle(title, fontsize=13)
    plt.savefig(os.path.join(OUT, filename), dpi=150, bbox_inches="tight")
    plt.close()


def make_galleries(test_ds, pred, true, conf):
    rng = np.random.default_rng(SEED)
    idxs = rng.choice(len(true), 24, replace=False)
    gallery(test_ds, idxs, pred, true, conf, "04_prediction_gallery.png",
            "Prediction gallery: 24 random test images (green = correct, red = wrong)")
    wrong = np.where(pred != true)[0]
    worst = wrong[np.argsort(-conf[wrong])][:12]
    gallery(test_ds, worst, pred, true, conf, "05_mistakes_gallery.png",
            "The 12 mistakes the model was MOST sure about")