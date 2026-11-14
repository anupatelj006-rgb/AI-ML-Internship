"""Load trained models and classify single images (with calibration, TTA, Grad-CAM)."""
import torch

import config
from data_loader import get_eval_transform, prepare_pil
from gradcam import gradcam, overlay
from models import build_model, gradcam_layer


def load_model(name: str, device="cpu"):
    path = config.MODEL_DIR / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run `python main.py` first.")
    ckpt = torch.load(path, map_location=device)
    model = build_model(name, len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


class Predictor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models, self.temps = {}, {}
        for name in config.MODEL_NAMES:
            if (config.MODEL_DIR / f"{name}.pt").exists():
                model, ckpt = load_model(name, self.device)
                self.models[name] = model
                self.temps[name] = float(ckpt.get("temperature", 1.0))
                self.classes = ckpt["classes"]
        if not self.models:
            raise SystemExit("No trained models found. Run `python main.py` first.")
        self.transform = get_eval_transform()   # same preprocessing as training/evaluation
        self.cam_name = next(n for n in ("transfer", "improved", "baseline") if n in self.models)

    def available(self):
        names = list(self.models)
        if len(names) > 1:
            names.append(config.ENSEMBLE)
        return names

    def _probs(self, name, x, tta):
        views = [x] + ([torch.flip(x, dims=[3])] if tta else [])
        logits = torch.stack([self.models[name](v) for v in views]).mean(0)
        return torch.softmax(logits / self.temps[name], dim=1)[0]

    @torch.no_grad()
    def predict(self, image, model_name, tta=True):
        img = prepare_pil(image)
        x = self.transform(img).unsqueeze(0).to(self.device)
        names = list(self.models) if model_name == config.ENSEMBLE else [model_name]
        probs = torch.stack([self._probs(n, x, tta) for n in names]).mean(0).cpu().tolist()
        scores = {c: float(p) for c, p in zip(self.classes, probs)}
        ordered = sorted(scores.values(), reverse=True)
        label = max(scores, key=scores.get)
        score, margin = ordered[0], ordered[0] - ordered[1]
        uncertain = score < config.CONFIDENCE_THRESHOLD or margin < config.MARGIN_THRESHOLD
        return {"scores": scores, "label": label, "score": score,
                "margin": margin, "uncertain": uncertain}

    def explain(self, image, class_name):
        model = self.models[self.cam_name]
        img = prepare_pil(image)
        x = self.transform(img).unsqueeze(0).to(self.device)
        cam, _ = gradcam(model, gradcam_layer(model, self.cam_name), x, self.classes.index(class_name))
        return overlay(img, cam)