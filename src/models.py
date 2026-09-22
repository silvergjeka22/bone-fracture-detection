"""The three classifiers compared in the project.

Layer names are kept identical to the original implementation, so weights trained
before the refactor (state dicts) still load.
"""

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from kymatio.torch import Scattering2D
from torchvision import models as tv_models

from .utils import count_parameters


class BoneFractureCNN(nn.Module):
    """4 conv blocks (32 -> 64 -> 128 -> 256, each conv + BN + ReLU + max-pool) and a 3-layer FC head."""

    def __init__(self, num_classes=2, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(dropout_rate)
        self.flatten_size = 256 * 14 * 14  # 224 / 2^4 = 14
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)


class BoneFractureResNet18(nn.Module):
    """ImageNet-pretrained ResNet18 with a new 2-class head (extra baseline, not required by the exam)."""

    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.5):
        super().__init__()
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = tv_models.resnet18(weights=weights)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.resnet(x)


class ScatNet2D(nn.Module):
    """Fixed wavelet scattering transform (Kymatio) followed by a trainable classifier.

    With J=4, L=8 each of the 3 input channels gives 1 + J*L + L^2*J*(J-1)/2 = 417
    scattering coefficients on a 14x14 grid (224 / 2^J), i.e. 3*417*14*14 = 245,196 features.
    """

    def __init__(self, num_classes=2, image_size=(224, 224), J=4, L=8,
                 hidden_dim=512, dropout_rate=0.5, in_channels=3):
        super().__init__()
        self.J, self.L = J, L
        self.scattering = Scattering2D(J=J, L=L, shape=image_size)  # no trainable parameters

        n_coeffs = 1 + J * L + L * L * J * (J - 1) // 2
        grid = (image_size[0] // 2 ** J) * (image_size[1] // 2 ** J)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * n_coeffs * grid, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.scattering(x)  # (B, 3, 417, 14, 14)
        return self.classifier(x.flatten(1))


def build_model(name, pretrained=True):
    """'cnn' | 'resnet18' | 'scatnet'. `pretrained` only matters for ResNet18."""
    if name == "cnn":
        return BoneFractureCNN()
    if name == "resnet18":
        return BoneFractureResNet18(pretrained=pretrained)
    if name == "scatnet":
        return ScatNet2D()
    raise ValueError(f"Unknown model '{name}'. Use 'cnn', 'resnet18' or 'scatnet'.")


def summary(names):
    """Trainable parameters of each model."""
    rows = []
    for name in names:
        model = build_model(name, pretrained=False)
        rows.append({"model": name, "class": type(model).__name__,
                     "trainable parameters": f"{count_parameters(model):,}"})
        del model
    return pd.DataFrame(rows).set_index("model")
