"""Dataset discovery, stratified splitting, and preprocessing transforms."""
import random

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_pil(img) -> Image.Image:
    """Shared first step for training images AND uploaded images:
    fix phone-camera rotation, force 3-channel RGB."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def load_image(path) -> Image.Image:
    with Image.open(path) as im:
        return prepare_pil(im)


def get_eval_transform():
    """The exact preprocessing used for validation, test AND uploaded images."""
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])


def get_train_transform(augment: bool):
    if not augment:
        return get_eval_transform()
    return transforms.Compose([
        transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])


def collect_samples():
    """Return a list of (path, label_index) for every image in config.CLASSES."""
    samples = []
    for idx, cls in enumerate(config.CLASSES):
        folder = config.DATA_DIR / cls
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Missing folder: {folder}\n"
                f"Expected layout: data/<class>/*.jpg for classes {config.CLASSES}.\n"
                "Run `python download_data.py` or see README.md.")
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)
        if not files:
            raise FileNotFoundError(
                f"No images found in {folder}\nRun `python download_data.py` or see README.md.")
        samples += [(str(p), idx) for p in files]
    return samples


def split_samples(samples):
    """Stratified train/val/test split (same split every run thanks to the seed)."""
    labels = [s[1] for s in samples]
    train, rest = train_test_split(
        samples, test_size=1 - config.SPLIT[0], stratify=labels, random_state=config.SEED)
    rest_labels = [s[1] for s in rest]
    val_share = config.SPLIT[1] / (config.SPLIT[1] + config.SPLIT[2])
    val, test = train_test_split(
        rest, train_size=val_share, stratify=rest_labels, random_state=config.SEED)
    return {"train": train, "val": val, "test": test}


class WasteDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(load_image(path)), label


def get_loaders(augment: bool):
    """augment=True applies augmentation to the TRAIN split only."""
    splits = split_samples(collect_samples())
    transform_for = {
        "train": get_train_transform(augment),
        "val": get_eval_transform(),
        "test": get_eval_transform(),
    }
    loaders = {}
    for name, samples in splits.items():
        loaders[name] = DataLoader(
            WasteDataset(samples, transform_for[name]),
            batch_size=config.BATCH_SIZE,
            shuffle=(name == "train"),
            num_workers=config.NUM_WORKERS,
        )
    return loaders, splits


def class_counts(samples):
    counts = {c: 0 for c in config.CLASSES}
    for _, label in samples:
        counts[config.CLASSES[label]] += 1
    return counts