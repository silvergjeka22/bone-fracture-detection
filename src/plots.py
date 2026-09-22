"""Every figure of the notebook.

Each function draws one figure, shows it and, if `save_to` is given, also writes it to disk
(for the report). Attribution overlays use one colour code for all methods so they can be
compared: red = evidence for the target class, blue = evidence against it.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from kymatio.scattering2d.filter_bank import filter_bank
from scipy.ndimage import gaussian_filter

from .data import denormalize


def _finish(fig, save_to=None):
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def show_images(images, labels, class_names, preds=None, title=None, save_to=None):
    """A row of X-rays titled with their class (and the prediction, red if wrong)."""
    fig, axes = plt.subplots(1, len(images), figsize=(2.2 * len(images), 2.6))
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.imshow(denormalize(images[i]), cmap="gray")
        text, color = class_names[labels[i]], "black"
        if preds is not None:
            text += f"\npred: {class_names[preds[i]]}"
            color = "green" if preds[i] == labels[i] else "red"
        ax.set_title(text, fontsize=9, color=color)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    _finish(fig, save_to)


def plot_cv_history(cv, save_to=None):
    """Learning curves per model: mean over folds (line) ± std (band), train vs validation."""
    fig, axes = plt.subplots(2, len(cv), figsize=(4.5 * len(cv), 6.5), squeeze=False)
    for col, (name, result) in enumerate(cv.items()):
        for row, metric in enumerate(["loss", "acc"]):
            ax = axes[row, col]
            for split, color in [("train", "tab:blue"), ("val", "tab:orange")]:
                curves = np.array([h[f"{split}_{metric}"] for h in result["histories"]])
                epochs = np.arange(1, curves.shape[1] + 1)
                mean, std = curves.mean(0), curves.std(0)
                ax.plot(epochs, mean, color=color, label=split)
                ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2)
            ax.set_title(f"{name}: {'loss' if metric == 'loss' else 'accuracy'}")
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
            ax.legend()
    _finish(fig, save_to)


def plot_confusion_matrices(test, class_names, save_to=None):
    """One confusion matrix per model (rows = true class, columns = predicted class)."""
    fig, axes = plt.subplots(1, len(test), figsize=(4 * len(test), 3.8), squeeze=False)
    for ax, (name, result) in zip(axes[0], test.items()):
        cm = np.array(result["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum():.1%})", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(len(class_names)), class_names)
        ax.set_yticks(range(len(class_names)), class_names, rotation=90, va="center")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{name}  (accuracy {result['accuracy']:.1%})")
    _finish(fig, save_to)


def plot_conv_filters(weight, title, n=32, save_to=None):
    """First-layer filters of a CNN.

    The input X-rays are grey (3 identical channels), so a filter acts on the image through the
    sum of its 3 channel kernels: that sum is what is shown (blue < 0 < red).
    """
    kernels = weight.detach().cpu().sum(1).numpy()[:n]
    cols = 8
    fig, axes = plt.subplots(int(np.ceil(len(kernels) / cols)), cols, figsize=(1.6 * cols, 1.6 * len(kernels) / cols + 0.6))
    for ax, k in zip(axes.flat, kernels):
        v = np.abs(k).max()
        ax.imshow(k, cmap="RdBu_r", vmin=-v, vmax=v)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(title)
    _finish(fig, save_to)


def plot_scattering_filters(J, L, save_to=None):
    """The fixed Morlet wavelets used by ScatNet (real part), one row per scale j, one column per angle.

    Taken from Kymatio's own filter bank. Each row is cropped around the centre to a size that
    grows with the scale (written on the left), otherwise the finest wavelets would be invisible.
    """
    size = 2 ** (J + 3)
    bank = filter_bank(size, size, J, L=L)
    fig, axes = plt.subplots(J, L, figsize=(1.6 * L, 1.7 * J), squeeze=False)
    for psi in bank["psi"]:
        j, theta = psi["j"], psi["theta"]
        spatial = np.fft.fftshift(np.fft.ifft2(psi["levels"][0])).real
        half = 3 * 2 ** j + 2
        crop = spatial[size // 2 - half:size // 2 + half + 1, size // 2 - half:size // 2 + half + 1]
        v = np.abs(crop).max()
        ax = axes[j, theta]
        ax.imshow(crop, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_title(f"{180 * theta / L:.0f}°", fontsize=9)
        if theta == 0:
            ax.set_ylabel(f"j={j}\n{2 * half + 1}px", fontsize=9)
    fig.suptitle(f"ScatNet: fixed Morlet wavelets (J={J} scales x L={L} orientations)")
    _finish(fig, save_to)


def _overlay(ax, image, attribution, smooth=1.5):
    """Grey X-ray + attribution. Colours are clipped at the 99th percentile of |attribution| and
    transparency follows |attribution|, so weak (noisy) values fade out instead of hiding the bone."""
    a = gaussian_filter(attribution, smooth) if smooth else attribution
    scale = np.percentile(np.abs(a), 99) or 1e-12
    a = np.clip(a / scale, -1, 1)
    ax.imshow(denormalize(image), cmap="gray")
    ax.imshow(a, cmap="bwr", vmin=-1, vmax=1, alpha=0.85 * np.abs(a))
    ax.axis("off")


def plot_attribution_grid(images, labels, maps, class_names, methods=None, preds=None,
                          indices=None, title=None, save_to=None):
    """Rows = images, columns = original X-ray + one overlay per XAI method."""
    methods = list(methods or maps)
    indices = list(range(len(images))) if indices is None else list(indices)
    fig, axes = plt.subplots(len(indices), len(methods) + 1,
                             figsize=(2.3 * (len(methods) + 1), 2.5 * len(indices)), squeeze=False)
    for row, i in enumerate(indices):
        ax = axes[row, 0]
        ax.imshow(denormalize(images[i]), cmap="gray")
        ax.axis("off")
        text, color = f"true: {class_names[labels[i]]}", "black"
        if preds is not None:
            text += f"\npred: {class_names[preds[i]]}"
            color = "green" if preds[i] == labels[i] else "red"
        ax.set_title(text, fontsize=9, color=color)
        for col, method in enumerate(methods, start=1):
            _overlay(axes[row, col], images[i], maps[method][i])
            if row == 0:
                axes[row, col].set_title(method, fontsize=10)
    if title:
        fig.suptitle(title, fontsize=13)
    _finish(fig, save_to)


def plot_scratch_vs_captum(images, attributions, index=0, save_to=None):
    """For one image: Captum's Occlusion, ours, and their absolute difference, one row per model."""
    fig, axes = plt.subplots(len(attributions), 4, figsize=(11, 2.8 * len(attributions)), squeeze=False)
    for row, (name, maps) in enumerate(attributions.items()):
        captum, ours = maps["Occlusion"][index], maps["Occlusion (scratch)"][index]
        v = np.abs(captum).max() or 1e-12
        panels = [(denormalize(images[index]), "gray", None, None, f"{name}: image"),
                  (captum, "bwr", -v, v, "Captum Occlusion"),
                  (ours, "bwr", -v, v, "Occlusion from scratch"),
                  (np.abs(captum - ours), "magma", 0, v, "|difference| (same scale)")]
        for ax, (img, cmap, vmin, vmax, text) in zip(axes[row], panels):
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(text, fontsize=10)
            ax.axis("off")
    _finish(fig, save_to)


def plot_agreement(matrices, save_to=None):
    """Method-agreement heatmaps (Spearman rho), one per model."""
    fig, axes = plt.subplots(1, len(matrices), figsize=(5.2 * len(matrices), 4.8), squeeze=False)
    for ax, (name, m) in zip(axes[0], matrices.items()):
        ax.imshow(m.values, cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(len(m)):
            for j in range(len(m)):
                ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(m)), m.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(m)), m.index, fontsize=8)
        ax.set_title(name)
    _finish(fig, save_to)
