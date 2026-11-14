"""Grad-CAM: highlights the image regions that drove the model's decision."""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from data_loader import get_eval_transform, load_image
from models import gradcam_layer


def gradcam(model, layer, x, class_idx=None):
    """x: (1,3,H,W) tensor. Returns (heatmap in [0,1] of size HxW, class index used)."""
    model.eval()
    store = {}

    def fwd(module, inputs, output):
        store["a"] = output.detach()
        output.register_hook(lambda g: store.__setitem__("g", g.detach()))

    handle = layer.register_forward_hook(fwd)
    x = x.clone().requires_grad_(True)
    with torch.enable_grad():
        logits = model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        model.zero_grad()
        logits[0, class_idx].backward()
    handle.remove()

    weights = store["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * store["a"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.cpu().numpy(), class_idx


def overlay(pil_img, cam, alpha=0.45, size=256):
    base = pil_img.convert("RGB").resize((size, size))
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    heat = matplotlib.colormaps["jet"](np.asarray(cam_img) / 255.0)[..., :3]
    heat = Image.fromarray((heat * 255).astype(np.uint8))
    return Image.blend(base, heat, alpha)


def gradcam_gallery(model, name, samples, y_true, y_pred, probs, idx, path, max_images=8):
    if len(idx) == 0:
        return None
    device = next(model.parameters()).device
    transform = get_eval_transform()
    layer = gradcam_layer(model, name)
    chosen = list(idx[:max_images])
    cols = 4
    rows = math.ceil(len(chosen) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, chosen):
        img = load_image(samples[i][0])
        x = transform(img).unsqueeze(0).to(device)
        cam, _ = gradcam(model, layer, x, int(y_pred[i]))
        ax.imshow(overlay(img, cam))
        ax.set_title(f"true: {config.CLASSES[y_true[i]]}\npred: {config.CLASSES[y_pred[i]]} "
                     f"({probs[i, y_pred[i]]:.0%})", fontsize=9)
    fig.suptitle(f"Grad-CAM on mistakes - {config.MODEL_TITLES[name]}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path