import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from config import DATA, OUT, CLASSES, GROUP_NAMES, CLASS_TO_GROUP, MEAN, STD, SEED


def show_samples():
    """One example of each clothing type, with its bigger category."""
    raw = datasets.FashionMNIST(DATA, train=True, download=True, transform=transforms.ToTensor())
    seen = {}
    for i in range(len(raw)):
        lab = int(raw.targets[i])
        if lab not in seen:
            seen[lab] = i
        if len(seen) == 10:
            break
    plt.figure(figsize=(12, 5))
    for k, lab in enumerate(sorted(seen)):
        plt.subplot(2, 5, k + 1)
        plt.imshow(raw[seen[lab]][0].squeeze(), cmap="gray")
        plt.title(f"{CLASSES[lab]}\n({GROUP_NAMES[CLASS_TO_GROUP[lab]]})", fontsize=9)
        plt.axis("off")
    plt.suptitle("Fashion-MNIST: one example of each class")
    plt.savefig(os.path.join(OUT, "01_samples.png"), dpi=150, bbox_inches="tight")
    plt.close()


def get_loaders(augment=False, batch_size=128):
    """Same train/validation split every time, so all models face the same exam."""
    normalize = transforms.Normalize((MEAN,), (STD,))
    plain = transforms.Compose([transforms.ToTensor(), normalize])
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ToTensor(), normalize])
    else:
        train_tf = plain

    train_full = datasets.FashionMNIST(DATA, train=True, download=True, transform=train_tf)
    val_full = datasets.FashionMNIST(DATA, train=True, download=True, transform=plain)
    test_ds = datasets.FashionMNIST(DATA, train=False, download=True, transform=plain)

    gen = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(60000, generator=gen).tolist()
    train_ds = Subset(train_full, perm[:54000])
    val_ds = Subset(val_full, perm[54000:])

    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size),
            DataLoader(test_ds, batch_size=batch_size),
            test_ds)