import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from config import DATA, OUT, CLASSES, MEAN, STD, SEED


def show_samples():
    """Step 1: load raw images, look at the tensor, and plot one of each class."""
    raw = datasets.FashionMNIST(DATA, train=True, download=True,
                                transform=transforms.ToTensor())
    img, label = raw[0]
    print("One image as a tensor -> shape:", tuple(img.shape),
          "| pixel range:", float(img.min()), "to", float(img.max()),
          "| label:", CLASSES[label])

    seen = {}
    for i in range(len(raw)):
        _, lab = raw[i]
        if lab not in seen:
            seen[lab] = i
        if len(seen) == 10:
            break

    plt.figure(figsize=(12, 5))
    for k, lab in enumerate(sorted(seen)):
        plt.subplot(2, 5, k + 1)
        plt.imshow(raw[seen[lab]][0].squeeze(), cmap="gray")
        plt.title(CLASSES[lab]); plt.axis("off")
    plt.suptitle("Fashion-MNIST: one example of each class")
    plt.savefig(os.path.join(OUT, "01_samples.png"), dpi=150, bbox_inches="tight")
    plt.close()


def get_loaders(batch_size=128):
    """Normalized datasets split into train / validation / test DataLoaders."""
    tf = transforms.Compose([
        transforms.ToTensor(),                   # picture -> tensor with values 0-1
        transforms.Normalize((MEAN,), (STD,)),   # centre the values around 0
    ])
    full_train = datasets.FashionMNIST(DATA, train=True, download=True, transform=tf)
    test_ds = datasets.FashionMNIST(DATA, train=False, download=True, transform=tf)

    gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full_train, [54000, 6000], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, val_loader, test_loader, test_ds