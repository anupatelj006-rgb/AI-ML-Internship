import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 28 -> 14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))   # 14 -> 7
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


class DeepCNN(nn.Module):
    """Three filter layers. BatchNorm keeps numbers steady, Dropout stops memorising."""
    def __init__(self, dropout=0.4):
        super().__init__()

        def block(c_in, c_out):
            return [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out),
                    nn.ReLU(), nn.MaxPool2d(2)]

        self.features = nn.Sequential(*block(1, 32), *block(32, 64), *block(64, 128))  # 28 -> 14 -> 7 -> 3
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 3 * 3, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(cfg):
    if cfg["model"] == "small":
        return SmallCNN(cfg["dropout"])
    return DeepCNN(cfg["dropout"])