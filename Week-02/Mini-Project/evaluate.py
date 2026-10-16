import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from config import OUT
from train import accuracy


def get_predictions(model, x_test):
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x_test), dim=1)   # outputs as percentages
    conf, pred = probs.max(1)
    return pred.numpy(), conf.numpy()


def report_test_accuracy(model, x_test, y_test):
    acc = accuracy(model, x_test, y_test)
    print(f"\nTest accuracy: {acc:.4f} ({acc*100:.2f}%)")
    return acc


def sample_predictions(x_test, y_true, pred, conf):
    plt.figure(figsize=(12, 5))
    for i in range(15):
        plt.subplot(3, 5, i + 1)
        plt.imshow(x_test[i], cmap="gray"); plt.axis("off")
        ok = pred[i] == y_true[i]
        plt.title(f"True {y_true[i]} | Pred {pred[i]}\n{conf[i]*100:.0f}% sure",
                  color="green" if ok else "red")
    plt.suptitle("Sample predictions (green = correct, red = wrong)")
    plt.savefig(os.path.join(OUT, "sample_predictions.png"), dpi=150, bbox_inches="tight")
    plt.close()


def confusion_and_report(y_true, pred):
    cm = confusion_matrix(y_true, pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("MNIST Confusion Matrix (Test Set)")
    plt.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("\n", classification_report(y_true, pred, digits=3))
    return cm


def wrong_predictions(x_test, y_true, pred, conf, cm):
    wrong = np.where(pred != y_true)[0]
    print(f"Total wrong: {len(wrong)} of {len(y_true)}")

    plt.figure(figsize=(14, 6))
    for k, i in enumerate(wrong[:10]):
        plt.subplot(2, 5, k + 1)
        plt.imshow(x_test[i], cmap="gray"); plt.axis("off")
        plt.title(f"True {y_true[i]} | Pred {pred[i]}\n{conf[i]*100:.0f}% sure", color="red")
    plt.suptitle("Incorrect predictions")
    plt.savefig(os.path.join(OUT, "wrong_predictions.png"), dpi=150, bbox_inches="tight")
    plt.close()

    errs = cm.copy()
    np.fill_diagonal(errs, 0)
    print("\nMost common mistakes:")
    for idx in np.argsort(errs.ravel())[::-1][:5]:
        t, p = divmod(idx, 10)
        print(f"  True {t} predicted as {p}: {errs[t, p]} times")