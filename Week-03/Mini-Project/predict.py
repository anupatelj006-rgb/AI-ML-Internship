import sys
import numpy as np
import torch
from PIL import Image, ImageOps
from config import DEVICE, MODEL_PATH
from model_io import load_model


def prepare(path, mean, std):
    img = Image.open(path).convert("L")                 # grey
    img = ImageOps.fit(img, (28, 28))                   # square crop, resize to 28x28
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.mean() > 0.5:                                # dataset items are light on dark
        arr = 1.0 - arr
    arr = (arr - mean) / std
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # shape 1 x 1 x 28 x 28


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/photo.jpg")
        return
    model, ckpt = load_model(MODEL_PATH)
    x = prepare(sys.argv[1], ckpt["mean"], ckpt["std"]).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top = probs.topk(3)
    print("Top 3 guesses:")
    for p, i in zip(top.values, top.indices):
        print(f"  {ckpt['classes'][int(i)]}: {p.item()*100:.1f}%")


if __name__ == "__main__":
    main()