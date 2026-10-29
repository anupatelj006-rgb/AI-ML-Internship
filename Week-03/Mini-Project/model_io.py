import torch
from config import DEVICE, CLASSES, MEAN, STD
from models import build_model


def save_model(model, name, path):
    torch.save({"model_name": name, "state_dict": model.state_dict(),
                "classes": CLASSES, "mean": MEAN, "std": STD}, path)


def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE)
    model = build_model(ckpt["model_name"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    return model, ckpt