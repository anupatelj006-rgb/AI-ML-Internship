import torch
from torchvision import datasets
from config import DATA


def load_data():
    """Returns (train, validation, test), each as (images, labels)."""
    train_ds = datasets.MNIST(DATA, train=True, download=True)
    test_ds = datasets.MNIST(DATA, train=False, download=True)

    x_all = train_ds.data.float() / 255.0     # pixels scaled to 0-1
    y_all = train_ds.targets
    x_test = test_ds.data.float() / 255.0
    y_test = test_ds.targets

    perm = torch.randperm(60000)
    tr, va = perm[:54000], perm[54000:]
    return (x_all[tr], y_all[tr]), (x_all[va], y_all[va]), (x_test, y_test)