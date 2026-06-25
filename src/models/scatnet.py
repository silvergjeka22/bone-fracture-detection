from statistics import mean

import torch
import torch.nn as nn
import torch.nn.functional as F
import kymatio

class BoneFractureScatNet(nn.Module):
    def __init__(
        self,
        num_classes=2,
        image_size=(224, 224),
        in_channels=3,
        J=4,
        L=8,
        hidden_dim=512,
        dropout_rate=0.5,
    ):
        super(BoneFractureScatNet, self).__init__()

        try:
            from kymatio.torch import Scattering2D
        except ImportError as exc:
            raise ImportError(
                "Kymatio is required for BoneFractureScatNet. "
                "Install it with: pip install kymatio"
            ) from exc

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if len(image_size) != 2:
            raise ValueError("image_size must be an int or a tuple/list (H, W)")

        self.image_size = tuple(image_size)
        self.in_channels = in_channels
        self.J = J
        self.L = L

        self.scattering = Scattering2D(J=self.J, shape=self.image_size, L=self.L)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, *self.image_size)
            n_coeff = self._scattering_per_channel(dummy).shape[2]

        self.n_coeff = n_coeff
        feature_dim = self.in_channels * self.n_coeff

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def _prepare_input(self, x):
        if x.ndim != 4:
            raise ValueError(
                f"Expected input with shape (B, C, H, W), got {tuple(x.shape)}"
            )

        if x.shape[1] != self.in_channels:
            if x.shape[1] == 1 and self.in_channels == 3:
                x = x.repeat(1, 3, 1, 1)
            elif x.shape[1] == 3 and self.in_channels == 1:
                x = x.mean(dim=1, keepdim=True)
            else:
                raise ValueError(
                    f"Expected {self.in_channels} input channels, got {x.shape[1]}"
                )

        if tuple(x.shape[-2:]) != self.image_size:
            x = F.interpolate(
                x, size=self.image_size, mode="bilinear", align_corners=False
            )

        return x

    def _scattering_per_channel(self, x):
        b, c, h, w = x.shape
        x_reshaped = x.reshape(b * c, h, w)
        s = self.scattering(x_reshaped)
        return s.reshape(b, c, s.shape[1], s.shape[2], s.shape[3])

    def forward(self, x):
        x = self._prepare_input(x)
        s = self._scattering_per_channel(x)  # (B, C, K, Hs, Ws)
        s = torch.log1p(torch.clamp(s, min=1e-6))  # clamp to prevent log(0)
        
        mean = s.mean(dim=(2,3,4), keepdim=True)
        std = s.std(dim=(2,3,4), keepdim=True) + 1e-5
        s = (s - mean) / std

        b, c, k, hs, ws = s.shape
        s = s.reshape(b, c * k, hs, ws)
        s = self.global_pool(s)
        s = s.reshape(b, -1)
        logits = self.classifier(s)
        return logits
