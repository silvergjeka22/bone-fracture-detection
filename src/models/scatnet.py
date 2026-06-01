import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from kymatio.torch import Scattering2D
    _KYMATIO_AVAILABLE = True
except ImportError:
    _KYMATIO_AVAILABLE = False
    raise ImportError("Kymatio is required. Install with: pip install kymatio")


def _generate_morlet_wavelets(J=2, L=8, size=64):
    """Generate theoretical 2D Morlet wavelets at J scales × L orientations."""
    filters = []
    x = np.linspace(-4, 4, size)
    y = np.linspace(-4, 4, size)
    X, Y = np.meshgrid(x, y)
    for j in range(J):
        scale = 2 ** j
        sigma = max(float(scale), 1.0)
        k0 = 5.0 / scale
        for l in range(L):
            angle = np.pi * l / L
            Xr = X * np.cos(angle) + Y * np.sin(angle)
            gaussian = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
            morlet = gaussian * np.cos(k0 * Xr)
            filters.append(morlet)
    return filters


class BoneFractureScatNet(nn.Module):
    """
    Scattering Network (ScatNet) for bone fracture detection.

    Uses Kymatio's Scattering2D (fixed Morlet wavelets) as feature extractor,
    followed by the same FC classifier head as BoneFractureCNN — only the
    number of input neurons to fc1 differs (exam requirement).

    Architecture:
        RGB input → grayscale → Scattering2D(J, L) → BN → ReLU
        → AdaptiveAvgPool(4,4) → Flatten
        → fc1(N→512) → ReLU → Dropout
        → fc2(512→128) → ReLU → Dropout
        → fc3(128→num_classes)
    """

    def __init__(self, num_classes=2, J=2, L=8, dropout_rate=0.5):
        super().__init__()
        self.J = J
        self.L = L

        # Fixed scattering transform — no learnable parameters
        self.scattering = Scattering2D(J=J, shape=(224, 224))

        # Determine actual output channels via a CPU dummy pass
        n_scat = self._probe_scat_channels()
        self._scat_channels = n_scat

        # BatchNorm on scattering coefficients
        self.bn_scat = nn.BatchNorm2d(n_scat)

        # Pool to fixed spatial size before FC layers
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        flatten_size = n_scat * 4 * 4

        # Dropout (same rate as CNN)
        self.dropout = nn.Dropout(dropout_rate)

        # Fully Connected Classifier — IDENTICAL TO CNN (only flatten_size differs)
        self.fc1 = nn.Linear(flatten_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def _probe_scat_channels(self):
        """Run a dummy forward pass to detect actual scattering output channels."""
        formula = 1 + self.J * self.L + (self.J * (self.J - 1) // 2) * self.L * self.L
        try:
            with torch.no_grad():
                dummy = torch.zeros(2, 224, 224)
                out = self.scattering(dummy)
                if out.dim() == 4:   # (B, C, H, W)
                    return out.shape[1]
                elif out.dim() == 5: # (B, 1, C, H, W) — older kymatio
                    return out.shape[2]
                elif out.dim() == 3: # (C, H, W) — single image mode
                    return out.shape[0]
        except Exception:
            pass
        return formula

    def forward(self, x):
        # x: (B, 3, 224, 224) — same interface as CNN

        # Convert RGB to grayscale: (B, 3, H, W) → (B, H, W)
        x_gray = x.mean(dim=1)

        # Scattering: (B, H, W) → (B, C_scat, H/2^J, W/2^J)
        x_scat = self.scattering(x_gray)

        # Normalise shape across kymatio versions
        if x_scat.dim() == 5:   # (B, 1, C, H', W')
            x_scat = x_scat.squeeze(1)
        elif x_scat.dim() == 3: # (C, H', W') — edge case
            x_scat = x_scat.unsqueeze(0)

        # BN + ReLU
        x_scat = self.bn_scat(x_scat)
        x_scat = F.relu(x_scat)

        # Pool → (B, n_scat, 4, 4)
        x_scat = self.pool(x_scat)

        # Flatten
        x_flat = x_scat.view(x_scat.size(0), -1)

        # FC classifier (same as CNN)
        x_flat = F.relu(self.fc1(x_flat))
        x_flat = self.dropout(x_flat)
        x_flat = F.relu(self.fc2(x_flat))
        x_flat = self.dropout(x_flat)
        return self.fc3(x_flat)

    def get_scatnet_filters(self):
        """
        Extract Morlet wavelet filters from the scattering transform.
        Returns a list of 2D numpy arrays (spatial domain).
        Falls back to theoretical Morlet wavelets if Kymatio internals
        cannot be accessed (version-dependent).
        """
        filters_out = []
        try:
            scat_cpu = self.scattering.cpu()
            psi_list = scat_cpu.psi
            for psi in psi_list:
                res_keys = sorted(k for k in psi.keys() if isinstance(k, int))
                if not res_keys:
                    continue
                f = psi[res_keys[0]]  # finest resolution
                if not isinstance(f, torch.Tensor):
                    continue
                if f.is_complex():
                    f_spatial = torch.fft.ifft2(f).real
                elif f.ndim >= 1 and f.shape[-1] == 2:
                    f_c = torch.view_as_complex(f.float().contiguous())
                    f_spatial = torch.fft.ifft2(f_c).real
                else:
                    f_c = torch.complex(f, torch.zeros_like(f))
                    f_spatial = torch.fft.ifft2(f_c).real
                filters_out.append(f_spatial.squeeze().cpu().numpy())
        except Exception:
            pass

        if not filters_out:
            filters_out = _generate_morlet_wavelets(J=self.J, L=self.L)

        return filters_out
