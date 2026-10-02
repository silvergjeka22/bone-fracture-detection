"""Small helpers: seed, device, parameter count, Kymatio/SciPy fix."""

import random

import numpy as np
import torch


def set_seed(seed=42):
    """Make runs repeatable (python, numpy and torch random generators)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """GPU if available, otherwise CPU."""
    if torch.cuda.is_available():
        try:
            torch.ones(1, device="cuda").add_(1)  
        except RuntimeError as error:
            raise RuntimeError(f"PyTorch cannot run on {torch.cuda.get_device_name(0)}: on Kaggle choose the "
                               "'GPU T4 x2' accelerator (kernel-metadata.json: machine_shape NvidiaTeslaT4).") from error
        torch.backends.cudnn.benchmark = True  # fixed input size: let cuDNN pick the fastest kernels
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def patch_scipy_for_kymatio():
    """Kymatio 0.3 imports scipy.special.sph_harm, removed in SciPy 1.17: put back an alias."""
    import scipy.special as special

    if not hasattr(special, "sph_harm"):
        special.sph_harm = lambda m, n, theta, phi: special.sph_harm_y(n, m, phi, theta)
