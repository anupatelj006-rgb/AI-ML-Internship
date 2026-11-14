import torch.nn as nn
from torchvision import models as tv_models

import config


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        side = config.IMG_SIZE // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * side * side, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out), nn.ReLU(),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out), nn.ReLU(),
        nn.MaxPool2d(2),
    )


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            _block(3, 32), _block(32, 64), _block(64, 128), _block(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class TransferNet(nn.Module):
    """ImageNet-pretrained ResNet18 with a new classification head."""

    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        self.pretrained_loaded = False
        backbone = None
        if pretrained:
            try:
                backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
                self.pretrained_loaded = True
            except Exception as e:
                print(f"WARNING: could not download pretrained weights ({e}). Training from scratch instead.")
        if backbone is None:
            backbone = tv_models.resnet18(weights=None)
        backbone.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, num_classes))
        self.backbone = backbone

    def set_backbone_trainable(self, flag: bool):
        for n, p in self.backbone.named_parameters():
            if not n.startswith("fc"):
                p.requires_grad = flag

    def forward(self, x):
        return self.backbone(x)


def build_model(name: str, num_classes: int = len(config.CLASSES), pretrained: bool = False):
    if name == "baseline":
        return BaselineCNN(num_classes)
    if name == "improved":
        return ImprovedCNN(num_classes)
    if name == "transfer":
        return TransferNet(num_classes, pretrained)
    raise ValueError(f"Unknown model: {name}")


def gradcam_layer(model, name: str):
    """Last convolutional feature map of each model (used by Grad-CAM)."""
    if name == "baseline":
        return model.features[4]
    if name == "improved":
        return model.features[3][5]
    if name == "transfer":
        return model.backbone.layer4
    raise ValueError(name)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())