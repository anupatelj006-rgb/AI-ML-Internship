import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from config import DATA, CLASSES, MEAN, STD, SEED


def get_data(batch_size=128):
    """Returns loaders, the test set, and the label arrays of each split."""
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
    full = datasets.FashionMNIST(DATA, train=True, download=True, transform=tf)
    test_ds = datasets.FashionMNIST(DATA, train=False, download=True, transform=tf)

    gen = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(60000, generator=gen)
    train_idx, val_idx = perm[:54000], perm[54000:]

    train_ds = Subset(full, train_idx.tolist())
    val_ds = Subset(full, val_idx.tolist())

    targets = {"Training": full.targets[train_idx].numpy(),
               "Validation": full.targets[val_idx].numpy(),
               "Test": test_ds.targets.numpy()}

    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size),
            DataLoader(test_ds, batch_size=batch_size),
            test_ds, targets)


def class_distribution(targets):
    """How many images of each class are in each split."""
    rows = []
    for c, name in enumerate(CLASSES):
        row = {"Class": name}
        for split, labels in targets.items():
            row[split] = int((labels == c).sum())
        row["Total"] = sum(row[s] for s in targets)
        rows.append(row)
    return pd.DataFrame(rows)