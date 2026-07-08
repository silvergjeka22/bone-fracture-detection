"""
Filter visualization utilities for CNN and ScatNet.

Provides:
  plot_cnn_filters        — visualise learned conv1 filters
  plot_scatnet_filters    — visualise Morlet wavelet filters (or theoretical fallback)
  plot_filter_comparison  — side-by-side CNN vs ScatNet
  plot_attribution_overlay— overlay single attribution on image
  plot_all_xai_methods    — grid of all 6 XAI methods overlaid on one image
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# CNN filter visualisation
# ---------------------------------------------------------------------------

def plot_cnn_filters(model, save_path=None, figsize=(16, 8)):
    """
    Visualise CNN conv1 filters (32 filters × 3 RGB channels each).
    Filters displayed as normalised RGB patches.
    """
    filters = model.get_conv1_filters().cpu().numpy()  # (32, 3, k, k)
    n = filters.shape[0]

    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min + 1e-8)

    fig, axes = plt.subplots(4, 8, figsize=figsize)
    fig.suptitle('CNN — Conv1 Learned Filters (32 filters, 3 channels)',
                 fontsize=14, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        if i < n:
            f_rgb = np.transpose(filters[i], (1, 2, 0))
            ax.imshow(np.clip(f_rgb, 0, 1), interpolation='nearest')
            ax.set_title(f'F{i+1}', fontsize=7)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# ScatNet / Morlet filter visualisation
# ---------------------------------------------------------------------------

def plot_scatnet_filters(model, save_path=None, figsize=(16, 5)):
    """
    Visualise ScatNet Morlet wavelet filters.
    Tries to extract actual filters from Kymatio; falls back to theoretical
    Morlet wavelets (mathematically equivalent) if internals are inaccessible.
    """
    filters = model.get_scatnet_filters()

    if not filters:
        _plot_theoretical_morlet(J=model.J, L=model.L,
                                 figsize=figsize, save_path=save_path)
        return

    # Display up to 16 filters
    n_show = min(16, len(filters))
    ncols = min(8, n_show)
    nrows = (n_show + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2 + 0.5))
    fig.suptitle('ScatNet — Morlet Wavelet Filters (from Kymatio)',
                 fontsize=14, fontweight='bold')
    axes_flat = np.array(axes).flatten()

    for i, ax in enumerate(axes_flat):
        if i < n_show:
            f = filters[i]
            if isinstance(f, torch.Tensor):
                f = f.cpu().numpy()
            f = np.real(f) if np.iscomplexobj(f) else np.array(f, dtype=float)
            if f.ndim > 2:
                f = f.squeeze()
            if f.ndim > 2:
                f = f[0]
            vmax = max(abs(f.min()), abs(f.max())) + 1e-8
            ax.imshow(f, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_title(f'ψ{i+1}', fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def _plot_theoretical_morlet(J=2, L=8, size=64, figsize=(16, 4), save_path=None):
    """Plot theoretical 2D Morlet wavelets at J scales × L orientations."""
    x = np.linspace(-4, 4, size)
    y = np.linspace(-4, 4, size)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(J, L, figsize=figsize)
    if J == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f'ScatNet — Morlet Wavelets  (J={J} scales, L={L} orientations)',
                 fontsize=14, fontweight='bold')

    for j in range(J):
        scale = 2 ** j
        sigma = max(float(scale), 1.0)
        k0 = 5.0 / scale
        for l in range(L):
            angle = np.pi * l / L
            Xr = X * np.cos(angle) + Y * np.sin(angle)
            morlet = np.exp(-(X**2 + Y**2) / (2 * sigma**2)) * np.cos(k0 * Xr)
            axes[j, l].imshow(morlet, cmap='RdBu_r', vmin=-1, vmax=1)
            axes[j, l].set_title(f'j={j}, {int(np.degrees(angle))}°', fontsize=7)
            axes[j, l].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Side-by-side CNN vs ScatNet comparison
# ---------------------------------------------------------------------------

def plot_filter_comparison(cnn_model, scatnet_model, save_path=None, figsize=(20, 8)):
    """
    Side-by-side comparison of CNN (learned) and ScatNet (fixed Morlet) filters.
    Top row: CNN conv1 filters.  Bottom row: theoretical Morlet wavelets.
    """
    cnn_filters = cnn_model.get_conv1_filters().cpu().numpy()
    f_min, f_max = cnn_filters.min(), cnn_filters.max()
    cnn_filters = (cnn_filters - f_min) / (f_max - f_min + 1e-8)

    J, L = scatnet_model.J, scatnet_model.L
    x = np.linspace(-4, 4, 64)
    X, Y = np.meshgrid(x, x)

    n_show = 16
    fig, axes = plt.subplots(2, n_show, figsize=figsize)
    fig.suptitle('Filter Comparison: CNN Learned Filters vs ScatNet Morlet Wavelets',
                 fontsize=14, fontweight='bold')

    for i in range(n_show):
        # CNN filter (top)
        if i < cnn_filters.shape[0]:
            f_rgb = np.transpose(cnn_filters[i], (1, 2, 0))
            axes[0, i].imshow(np.clip(f_rgb, 0, 1), interpolation='nearest')
        axes[0, i].set_title(f'CNN F{i+1}', fontsize=6)
        axes[0, i].axis('off')

        # Morlet wavelet (bottom)
        j = i // L
        l = i % L
        if j < J:
            scale = 2 ** j
            k0 = 5.0 / scale
            sigma = max(float(scale), 1.0)
            angle = np.pi * l / L
            Xr = X * np.cos(angle) + Y * np.sin(angle)
            morlet = np.exp(-(X**2 + Y**2) / (2 * sigma**2)) * np.cos(k0 * Xr)
            axes[1, i].imshow(morlet, cmap='RdBu_r', vmin=-1, vmax=1)
            axes[1, i].set_title(f'ψ j={j},l={l}', fontsize=6)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('CNN\n(learned)', fontsize=9, rotation=0, labelpad=40, va='center')
    axes[1, 0].set_ylabel('ScatNet\n(fixed)', fontsize=9, rotation=0, labelpad=40, va='center')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Attribution overlay utilities
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def _denorm(image_tensor):
    """Denormalise an ImageNet-normalised (C,H,W) tensor to a [0,1] HWC array."""
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(img, 0, 1)


def _norm01(attr):
    """Normalise attribution to [0, 1]."""
    a = np.array(attr, dtype=float)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-8)


def plot_attribution_overlay(image_tensor, attribution_map, title='Attribution',
                              alpha=0.5, cmap='hot', figsize=(12, 4), save_path=None):
    """
    Show original | heatmap | overlay in a single row.
    """
    img = _denorm(image_tensor)
    attr = _norm01(attribution_map)
    if attr.shape != img.shape[:2]:
        attr = attr.squeeze()

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(img);   axes[0].set_title('Original Image');       axes[0].axis('off')
    axes[1].imshow(attr, cmap=cmap); axes[1].set_title('Attribution Heatmap'); axes[1].axis('off')
    axes[2].imshow(img); axes[2].imshow(attr, cmap=cmap, alpha=alpha)
    axes[2].set_title(f'{title} Overlay'); axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_all_xai_methods(image_tensor, attributions_dict, model_name='CNN',
                          class_name='Fractured', alpha=0.5, cmap='hot',
                          figsize=(18, 9), save_path=None):
    """
    Plot all 6 XAI method attributions overlaid on the original image in a grid.

    Args:
        image_tensor: (C, H, W) normalised tensor (no batch dim)
        attributions_dict: ordered dict  method_name → (H,W) numpy array or None
        model_name: 'CNN' or 'ScatNet'
        class_name: 'Fractured' or 'Not Fractured'
        alpha: overlay transparency
        cmap: heatmap colormap
    """
    img = _denorm(image_tensor)
    methods = list(attributions_dict.keys())
    n_total = len(methods) + 1   # +1 for original
    n_cols = 4
    n_rows = (n_total + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(f'XAI Methods — {model_name} — Class: {class_name}',
                 fontsize=14, fontweight='bold')
    axes_flat = axes.flatten()

    # Cell 0: original image
    axes_flat[0].imshow(img)
    axes_flat[0].set_title('Original Image', fontweight='bold', fontsize=10)
    axes_flat[0].axis('off')

    for i, (method_name, attr) in enumerate(attributions_dict.items()):
        ax = axes_flat[i + 1]
        if attr is None:
            ax.imshow(img, alpha=0.4)
            ax.text(0.5, 0.5, f'{method_name}\n(N/A — {model_name})',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.85))
        else:
            attr_n = _norm01(attr)
            ax.imshow(img)
            ax.imshow(attr_n, cmap=cmap, alpha=alpha)
        ax.set_title(method_name, fontsize=10, fontweight='bold')
        ax.axis('off')

    for j in range(len(methods) + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_triplet(method_network, title, inputs, attribution_all):
    fig, axes = plt.subplots(8, 3, figsize=(12, 24))

    for i in range(8):
        img = inputs[i].mean(0).cpu().numpy()
        sal = np.abs(attribution_all.get(method_network)[i])
        sal = gaussian_filter(sal, sigma=1.5)
        
        # Normalizza 0-1
        sal = (sal - sal.min()) / (sal.max() - sal.min())
        
        # Colonna 1: immagine originale
        axes[i, 0].imshow(img, cmap='gray')
        # if i == 0: axes[i, 0].set_title('Original', fontsize=15)
        axes[i, 0].axis('off')
        
        # Colonna 2: solo saliency
        axes[i, 1].imshow(sal, cmap='hot')
        if i == 0: axes[i, 1].set_title(title, fontsize=15)
        axes[i, 1].axis('off')
        
        # Colonna 3: overlay
        axes[i, 2].imshow(img, cmap='gray')
        axes[i, 2].imshow(sal, cmap='hot', alpha=0.5,
                        vmin=np.percentile(sal, 80), vmax=sal.max())
        # axes[i, 2].set_title(f'Overlay - {labels[i]}', fontsize=8)
        axes[i, 2].axis('off')

    plt.tight_layout()
