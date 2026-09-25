"""The models compared in the project.

Every model = feature extractor + the SAME `Classifier` (exam requirement: only the number of
input neurons may differ). Input: one grey channel, 224x224, normalised to [-1, 1].

    cnn       4 learned conv blocks                       -> 256 maps of 14x14 -> Classifier(50,176)
    scatnet   fixed wavelet scattering (Kymatio, J=4, L=8) -> 417 maps of 14x14 -> Classifier(81,732)
    resnet18  ImageNet-pretrained ResNet18 (fine-tuned)   -> 512 features      -> Classifier(512)

CNN and ScatNet both end on a 14x14 grid (224 / 2^4): the CNN through 4 max-poolings, ScatNet
through its low-pass averaging at scale 2^J = 16 pixels, so their classifiers see the same layout.
"""

import pandas as pd
import torch
import torch.nn as nn
from torchvision import models as tv_models

from .utils import count_parameters, patch_scipy_for_kymatio

patch_scipy_for_kymatio()
from kymatio.torch import Scattering2D  # noqa: E402  (needs the patch above)

MODEL_NAMES = ("cnn", "scatnet", "resnet18")


class Classifier(nn.Sequential):
    """The final classifier shared by all models: flatten -> 512 -> 128 -> 2 with ReLU and dropout."""

    def __init__(self, in_features, num_classes=2, dropout=0.5):
        super().__init__(
            nn.Flatten(),
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )


def conv_block(in_channels, out_channels, kernel_size=3):
    """conv -> batch norm -> ReLU -> 2x2 max-pool (halves the resolution)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class BoneFractureCNN(nn.Module):
    """4 conv blocks with 32 -> 64 -> 128 -> 256 filters.

    The first layer uses 7x7 filters (the others 3x3): large enough to show clear edge/texture
    detectors, which we compare with the ScatNet wavelets.
    """

    def __init__(self, image_size=224, num_classes=2, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32, kernel_size=7),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        grid = image_size // 16
        self.classifier = Classifier(256 * grid * grid, num_classes, dropout)

    def forward(self, x):
        return self.classifier(self.features(x))

    @property
    def first_conv(self):
        return self.features[0][0]

    @property
    def cam_layer(self):
        """Last convolutional block (256 x 14 x 14): the layer Grad-CAM explains."""
        return self.features[-1]


class ScatNet(nn.Module):
    """Wavelet scattering transform (fixed, nothing learned) + the shared classifier.

    Scattering2D(J, L) filters the image with Morlet wavelets at J scales and L orientations,
    takes the modulus, repeats once (2nd order) and averages every map over 2^J pixels:
    1 + J*L + L^2*J*(J-1)/2 = 417 coefficient maps of 14x14 for J=4, L=8.
    The coefficients are log-compressed and batch-normalised (their magnitudes span several
    orders), which is the only trainable part besides the classifier.
    """

    def __init__(self, image_size=224, J=4, L=8, num_classes=2, dropout=0.5):
        super().__init__()
        self.J, self.L = J, L
        self.scattering = Scattering2D(J=J, shape=(image_size, image_size), L=L)
        self.n_coefficients = 1 + J * L + L * L * J * (J - 1) // 2
        self.norm = nn.BatchNorm2d(self.n_coefficients)
        grid = image_size // 2 ** J
        self.classifier = Classifier(self.n_coefficients * grid * grid, num_classes, dropout)

    def features(self, x):
        s = self.scattering(x * 0.5 + 0.5)  # back to [0, 1] so that every coefficient is >= 0
        s = s.flatten(1, 2)                   # (B, 1, K, 14, 14) -> (B, K, 14, 14)
        return self.norm(torch.log(s + 1e-3))

    def forward(self, x):
        return self.classifier(self.features(x))


class ResNet18(nn.Module):
    """ImageNet-pretrained ResNet18, fine-tuned; its 1000-class layer is replaced by the shared classifier."""

    def __init__(self, num_classes=2, dropout=0.5, pretrained=True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = tv_models.resnet18(weights=weights)
        self.backbone.fc = nn.Identity()
        for module in self.backbone.modules():  # in-place ReLUs break the backward hooks of the XAI methods
            if isinstance(module, nn.ReLU):
                module.inplace = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.classifier = Classifier(512, num_classes, dropout)

    def forward(self, x):
        x = (x * 0.5 + 0.5).expand(-1, 3, -1, -1)  # grey [-1, 1] -> RGB [0, 1]
        return self.classifier(self.backbone((x - self.mean) / self.std))

    @property
    def first_conv(self):
        return self.backbone.conv1

    @property
    def cam_layer(self):
        return self.backbone.layer4


def build_model(name, image_size=224, pretrained=True):
    """'cnn' | 'scatnet' | 'resnet18'. `pretrained` only matters for ResNet18."""
    if name == "cnn":
        return BoneFractureCNN(image_size)
    if name == "scatnet":
        return ScatNet(image_size)
    if name == "resnet18":
        return ResNet18(pretrained=pretrained)
    raise ValueError(f"Unknown model '{name}'. Use one of {MODEL_NAMES}.")


def summary(names, image_size=224):
    """Feature extractor, classifier input size and trainable parameters of each model."""
    extractor = {"cnn": "4 learned conv blocks (7x7, 3x3, 3x3, 3x3)",
                 "scatnet": "fixed scattering, J=4 scales x L=8 angles",
                 "resnet18": "ResNet18, ImageNet-pretrained"}
    rows = {}
    for name in names:
        model = build_model(name, image_size, pretrained=False)
        head = count_parameters(model.classifier)
        rows[name] = {"feature extractor": extractor[name],
                      "classifier input": model.classifier[1].in_features,
                      "feature-extractor parameters": f"{count_parameters(model) - head:,}",
                      "classifier parameters": f"{head:,}",
                      "total": f"{count_parameters(model):,}"}
    return pd.DataFrame(rows).T


def classifier_layout(model):
    """The classifier's layers with their sizes, input size removed: identical for every model."""
    layout = []
    for layer in model.classifier:
        if isinstance(layer, nn.Linear):
            layout.append(("Linear", "in" if layer is model.classifier[1] else layer.in_features, layer.out_features))
        else:
            layout.append((type(layer).__name__,))
    return layout
