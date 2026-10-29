import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
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


def plot_curves(histories):
    colors = {"Fully connected": "tab:orange", "CNN": "tab:blue"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, h in histories.items():
        c = colors.get(name)
        ax[0].plot(h["train_acc"], color=c, label=f"{name} - training")
        ax[0].plot(h["val_acc"], "--", color=c, label=f"{name} - validation")
        ax[1].plot(h["train_loss"], color=c, label=f"{name} - training")
        ax[1].plot(h["val_loss"], "--", color=c, label=f"{name} - validation")
    ax[0].set_title("Accuracy"); ax[1].set_title("Loss")
    for a in ax:
        a.set_xlabel("Epoch"); a.legend()
    plt.savefig(os.path.join(OUT, "02_training_validation_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


def confusion_and_report(true, pred, title, filename):
    cm = confusion_matrix(true, pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues",
                                                           colorbar=False, xticks_rotation=45)
    plt.title(title)
    plt.savefig(os.path.join(OUT, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n{title}\n", classification_report(true, pred, target_names=CLASSES, digits=3))
    return cm


def show_misclassified(test_ds, pred, true, conf, n=5):
    """Shows the n wrong answers the CNN was MOST sure about."""
    wrong = np.where(pred != true)[0]
    order = wrong[np.argsort(-conf[wrong])][:n]

    fig, ax = plt.subplots(1, n, figsize=(3.2 * n, 4))
    print("\nMisclassified examples:")
    for a, i in zip(ax, order):
        img, _ = test_ds[i]
        a.imshow(img.squeeze() * STD + MEAN, cmap="gray"); a.axis("off")
        a.set_title(f"True: {CLASSES[true[i]]}\nPred: {CLASSES[pred[i]]}\n{conf[i]*100:.0f}% sure",
                    color="red", fontsize=9)
        print(f"  Test image #{i}: true = {CLASSES[true[i]]}, predicted = {CLASSES[pred[i]]}, "
              f"confidence = {conf[i]*100:.0f}%")
    plt.savefig(os.path.join(OUT, "04_misclassified.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_filters_and_maps(cnn, test_ds, idx=0):
    """What the first 16 filters look like, and what they 'see' in one image."""
    cnn.eval()
    w = cnn.features[0].weight.detach().cpu()          # 16 filters of size 3x3
    img, label = test_ds[idx]
    with torch.no_grad():
        maps = cnn.features[1](cnn.features[0](img.unsqueeze(0).to(DEVICE))).squeeze(0).cpu()

    fig, ax = plt.subplots(3, 8, figsize=(14, 6.5))
    for k in range(16):
        a = ax[k // 8][k % 8]
        a.imshow(w[k, 0], cmap="RdBu"); a.axis("off"); a.set_title(f"Filter {k}", fontsize=7)
    ax[2][0].imshow(img.squeeze() * STD + MEAN, cmap="gray")
    ax[2][0].set_title(CLASSES[label], fontsize=7)
    for k in range(7):
        ax[2][k + 1].imshow(maps[k], cmap="gray"); ax[2][k + 1].set_title(f"Map {k}", fontsize=7)
    for a in ax[2]:
        a.axis("off")
    plt.suptitle("Top: 16 learned 3x3 filters | Bottom: original image and what filters 0-6 found")
    plt.savefig(os.path.join(OUT, "05_filters_and_feature_maps.png"), dpi=150, bbox_inches="tight")
    plt.close()