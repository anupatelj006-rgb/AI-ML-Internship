import torch.nn as nn


class FullyConnected(nn.Module):
    """Baseline: sees every pixel separately, with no idea who the neighbours are."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                 # 1x28x28 -> 784 numbers in a row
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),           # one score per clothing type
        )

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """Looks at small neighbourhoods of pixels, then zooms out."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # 16 filters, 28x28 stays 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                              # zoom out: 28x28 -> 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 32 filters on the 16 maps
            nn.ReLU(),
            nn.MaxPool2d(2),                              # zoom out: 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # 32 x 7 x 7 = 1568 numbers
            nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
            nn.Dropout(0.3),                              # stops memorising
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))