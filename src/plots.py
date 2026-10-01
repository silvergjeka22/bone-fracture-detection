"""Every figure, one function each; fixed colours per model, class, method and box."""

import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

from .data import denormalize
from .localize import evidence, heatmap_to_box, iou, pointing_game
from .utils import patch_scipy_for_kymatio

patch_scipy_for_kymatio()
from kymatio.scattering2d.filter_bank import filter_bank  # noqa: E402

# ----------------------------------------------------------------------------- style
INK, INK_2, GRID, SURFACE, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#9a998f"
MODEL_COLORS = {"cnn": "#2a78d6", "scatnet": "#eb6834", "resnet18": "#1baf7a"}
MODEL_LABELS = {"cnn": "CNN", "scatnet": "ScatNet", "resnet18": "ResNet18"}
CLASS_COLORS = ["#4a3aa7", "#eda100"]  # fractured, not fractured
METHOD_COLORS = {"Saliency": "#2a78d6", "Integrated Gradients": "#eb6834", "Guided Backprop": "#1baf7a",
                 "Grad-CAM": "#eda100", "Occlusion": "#e87ba4", "LIME": "#008300",
                 "Occlusion (ours)": "#4a3aa7", "Random": NEUTRAL}
BOX_COLORS = {"radiologist": "#00c2d1", "ours": "#ffd23f", "detector": "#ff4fd8"}  # boxes drawn on the X-rays
DIVERGING = LinearSegmentedColormap.from_list("evidence", ["#1c5cab", "#2a78d6", "#f0efec", "#e34948", "#a8211f"])
POSITIVE = LinearSegmentedColormap.from_list("for", ["#f0efec", "#e34948", "#a8211f"])  # evidence for the class
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#f0efec", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2, "font.size": 10,
})


def label(name):
    """'resnet18_box_contrast' -> 'ResNet18 + box + contrast'."""
    base, *extra = name.split("_")
    return " + ".join([MODEL_LABELS.get(base, base), *extra]) if base in MODEL_LABELS else name


def _color(name):
    """Fixed colour of a model; the guided versions get violet (box) and pink (box + contrast)."""
    if name in MODEL_COLORS:
        return MODEL_COLORS[name]
    return "#e87ba4" if name.endswith("contrast") else "#4a3aa7"


def _rgba(values, cmap, alpha):
    """Colour image with a per-pixel transparency (works on every matplotlib version)."""
    rgba = cmap(np.clip(values, 0, 1))
    rgba[..., 3] = np.clip(alpha, 0, 1)
    return rgba


def _finish(fig, save_to=None):
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def _grid_axes(ax, axis="y"):
    ax.grid(True, axis=axis)
    ax.set_axisbelow(True)


def _draw_box(ax, box, who):
    """A fracture box (x0, y0, x1, y1) in pixel coordinates: the radiologist's dashed, ours and YOLO's solid."""
    x0, y0, x1, y1 = box
    ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, linewidth=1.6,
                           edgecolor=BOX_COLORS[who], linestyle="--" if who == "radiologist" else "-"))


# ----------------------------------------------------------------------------- data

def show_samples(image_set, n_per_class=5, seed=0, title=None, save_to=None):
    """A row of random images per class, with the radiologists' fracture boxes."""
    rng = np.random.default_rng(seed)
    classes = image_set.class_names
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(2.1 * n_per_class, 2.3 * len(classes)), squeeze=False)
    for c, name in enumerate(classes):
        idx = rng.choice(np.flatnonzero(image_set.labels == c), n_per_class, replace=False)
        for ax, i in zip(axes[c], idx):
            ax.imshow(image_set.images[i], cmap="gray", vmin=0, vmax=255)
            for box in image_set.boxes[i]:
                _draw_box(ax, box, "radiologist")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[c, 0].set_ylabel(name, color=CLASS_COLORS[c], fontsize=11)
    if title:
        fig.suptitle(title)
    _finish(fig, save_to)


def plot_dataset_overview(sets, save_to=None):
    """Images per split and class, and the original image size in every split."""
    splits = list(sets)
    classes = sets[splits[0]].class_names
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
    width = 0.38
    for c, name in enumerate(classes):
        counts = [int((sets[s].labels == c).sum()) for s in splits]
        x = np.arange(len(splits)) + (c - 0.5) * (width + 0.02)
        ax1.bar(x, counts, width, color=CLASS_COLORS[c], label=name)
        for xi, n in zip(x, counts):
            ax1.text(xi, n, f"{n:,}", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax1.set_xticks(range(len(splits)), splits)
    ax1.set_ylabel("images")
    ax1.set_title("Images per split and class")
    ax1.legend()
    _grid_axes(ax1)

    sizes = [np.sqrt(sets[s].sizes[:, 0] * sets[s].sizes[:, 1]) for s in splits]
    parts = ax2.boxplot(sizes, vert=False, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, linewidth=1.5))
    for box in parts["boxes"]:
        box.set(facecolor="#cde2fb", edgecolor="#2a78d6")
    for key in ("whiskers", "caps"):
        for line in parts[key]:
            line.set(color="#2a78d6")
    ax2.set_yticks(range(1, len(splits) + 1), splits)
    ax2.set_xscale("log")
    ax2.set_xlabel("original size: sqrt(width x height) in pixels (log scale)")
    ax2.set_title("Original image size per split")
    _grid_axes(ax2, "x")
    ax2.invert_yaxis()
    _finish(fig, save_to)


def show_duplicate_pairs(image_set, pairs, n=6, title=None, save_to=None):
    """The n most similar pairs of near-duplicates: one image above, its copy below."""
    pairs = pairs.sort_values("similarity", ascending=False).head(n)
    if len(pairs) == 0:
        print("no near-duplicates found")
        return
    fig, axes = plt.subplots(2, len(pairs), figsize=(2.1 * len(pairs), 4.6), squeeze=False)
    for k, (_, row) in enumerate(pairs.iterrows()):
        for r, idx in enumerate([int(row["i"]), int(row["j"])]):
            ax = axes[r, k]
            ax.imshow(image_set.images[idx], cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            ax.text(0.03, 0.97, image_set.class_names[image_set.labels[idx]], transform=ax.transAxes,
                    ha="left", va="top", fontsize=7.5, color=INK,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=1.5))
        axes[1, k].text(0.5, -0.08, f"similarity {row['similarity']:.3f}", transform=axes[1, k].transAxes,
                        ha="center", va="top", fontsize=8, color=INK_2)
    if title:
        fig.suptitle(title)
    _finish(fig, save_to)


# ----------------------------------------------------------------------------- training and results

def plot_learning_curves(cv, final=None, save_to=None):
    """Train (grey) and validation (model colour) on the same axes: CV mean ± std, and the final training."""
    names = list(cv)
    cols = 4 if final else 2
    fig, axes = plt.subplots(len(names), cols, figsize=(3.6 * cols, 2.7 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        color = _color(name)
        panels = [("cv", "loss"), ("cv", "acc")] + ([("final", "loss"), ("final", "acc")] if final else [])
        for col, (source, metric) in enumerate(panels):
            ax = axes[row, col]
            for split, c in [("train", NEUTRAL), ("val", color)]:
                if source == "cv":
                    curves = np.array([h[f"{split}_{metric}"] for h in cv[name]["histories"]])
                    mean, std = curves.mean(0), curves.std(0)
                else:
                    mean = np.array(final[name]["history"][f"{split}_{metric}"])
                    std = np.zeros_like(mean)
                epochs = np.arange(1, len(mean) + 1)
                ax.plot(epochs, mean, color=c, label="train" if split == "train" else "validation",
                        marker="o" if len(mean) < 4 else None, markersize=5)
                ax.fill_between(epochs, mean - std, mean + std, color=c, alpha=0.18, linewidth=0)
            if source == "final":
                ax.axvline(final[name]["best_epoch"], color=INK_2, linewidth=0.8)
            where = f"{len(cv[name]['histories'])}-fold CV (mean ± std)" if source == "cv" else "final training"
            ax.set_title(f"{label(name)}: {'loss' if metric == 'loss' else 'accuracy'}, {where}", fontsize=9.5)
            ax.set_xlabel("epoch")
            _grid_axes(ax, "both")
            if row == 0 and col == 0:
                ax.legend()
    _finish(fig, save_to)


def plot_model_comparison(table, best=None, save_to=None):
    """Scores of every model (dots, model colours) for each metric, on one percentage axis."""
    metrics = ["CV accuracy", "CV F1", "test accuracy", "test F1", "test recall", "test AUC"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for k, name in enumerate(table.index):
        y = np.arange(len(metrics)) + (k - (len(table) - 1) / 2) * 0.18
        text = label(name) + ("  (best, chosen by CV)" if name == best else "")
        ax.scatter(100 * table.loc[name, metrics].astype(float), y, s=60, color=_color(name),
                   edgecolor=SURFACE, linewidth=1.5, zorder=3, label=text)
    ax.set_yticks(range(len(metrics)), metrics)
    ax.invert_yaxis()
    ax.set_xlabel("score (%)")
    ax.set_title("Model comparison: cross-validation (training split) and test split")
    _grid_axes(ax, "x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    _finish(fig, save_to)


def plot_confusion_matrices(test, class_names, save_to=None):
    """One confusion matrix per model (rows = true class, columns = predicted class)."""
    fig, axes = plt.subplots(1, len(test), figsize=(3.9 * len(test), 3.6), squeeze=False)
    for ax, (name, result) in zip(axes[0], test.items()):
        cm = np.array(result["confusion_matrix"])
        ax.imshow(cm, cmap=SEQUENTIAL, vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n{cm[i, j] / max(cm[i].sum(), 1):.1%}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() * 0.6 else INK, fontsize=10)
        ax.set_xticks(range(2), class_names)
        ax.set_yticks(range(2), class_names, rotation=90, va="center")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{label(name)}: accuracy {result['accuracy']:.1%}")
        for spine in ax.spines.values():
            spine.set_visible(False)
    _finish(fig, save_to)


def plot_roc(test, positive_name="fractured", save_to=None):
    """ROC curve of every model for detecting the positive class."""
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    for name, r in test.items():
        fpr, tpr, _ = roc_curve(np.array(r["y_true"]) == 0, r["prob_positive"])
        ax.plot(fpr, tpr, color=_color(name), label=f"{label(name)} (AUC {r['auc']:.3f})")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel(f"true positive rate ({positive_name})")
    ax.set_title("ROC on the test split")
    ax.set_aspect("equal")
    _grid_axes(ax, "both")
    ax.legend(loc="lower right")
    _finish(fig, save_to)


# ----------------------------------------------------------------------------- filters

def _filter_panel(ax, kernel):
    v = np.abs(kernel).max() or 1.0
    ax.imshow(kernel, cmap=DIVERGING, vmin=-v, vmax=v)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_conv_filters(weight, title, n=32, save_to=None):
    """First-layer filters of a CNN (summed over input channels; blue < 0 < red)."""
    kernels = weight.detach().cpu().float().sum(1).numpy()[:n]
    cols = 8
    rows = int(np.ceil(len(kernels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.3 * cols, 1.3 * rows + 0.5), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, k in zip(axes.flat, kernels):
        ax.axis("on")
        _filter_panel(ax, k)
    fig.suptitle(title)
    _finish(fig, save_to)


def morlet_wavelets(J, L, size=None):
    """Real part of Kymatio's Morlet wavelets in space, cropped per scale: {(j, theta): 2D array}."""
    size = size or 2 ** (J + 3)
    bank = filter_bank(size, size, J, L=L)
    out = {}
    for psi in bank["psi"]:
        j, theta = psi["j"], psi["theta"]
        spatial = np.fft.fftshift(np.fft.ifft2(psi["levels"][0])).real
        half = 3 * 2 ** j + 2
        out[(j, theta)] = spatial[size // 2 - half:size // 2 + half + 1, size // 2 - half:size // 2 + half + 1]
    return out


def stripe_angle(kernel):
    """Orientation (degrees, 0 = horizontal) of the stripes of an oriented filter, from its spectrum peak."""
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(kernel, (64, 64))))
    row, col = np.unravel_index(spectrum.argmax(), spectrum.shape)
    wave = np.degrees(np.arctan2(32 - row, col - 32))  # direction of oscillation (image y axis points up)
    return (wave + 90) % 180                           # stripes are perpendicular to it


def plot_scattering_filters(J, L, save_to=None):
    """ScatNet's fixed Morlet wavelets (real part): one row per scale j, one column per orientation."""
    wavelets = morlet_wavelets(J, L)
    step = 180 / L  # the spectrum peak is coarse: snap to the nearest of the L orientations
    angles = {t: (round(stripe_angle(wavelets[(J - 1, t)]) / step) * step) % 180 for t in range(L)}
    order = sorted(range(L), key=lambda t: angles[t])  # columns sorted by stripe orientation
    fig, axes = plt.subplots(J, L, figsize=(1.3 * L, 1.4 * J + 0.4), squeeze=False)
    for (j, theta), kernel in wavelets.items():
        col = order.index(theta)
        _filter_panel(axes[j, col], kernel)
        if j == 0:
            axes[j, col].set_title(f"{angles[theta]:.0f}°", fontsize=9)
        if col == 0:
            axes[j, col].set_ylabel(f"j={j}\n{kernel.shape[0]}px", fontsize=9)
    fig.suptitle(f"ScatNet: fixed Morlet wavelets (J={J} scales x L={L} orientations)")
    _finish(fig, save_to)


def plot_frequency_coverage(weight, J, L, size=64, save_to=None):
    """Which spatial frequencies each model looks at: learned CNN filters vs the wavelet tiling."""
    kernels = weight.detach().cpu().float().sum(1).numpy()
    cnn = np.zeros((size, size))
    for k in kernels:
        power = np.abs(np.fft.fftshift(np.fft.fft2(k, (size, size)))) ** 2
        cnn += power / (power.max() or 1)
    bank = filter_bank(size, size, J, L=L)
    wav = np.zeros((size, size))
    for psi in bank["psi"]:
        power = np.fft.fftshift(np.abs(psi["levels"][0])) ** 2
        wav += power / (power.max() or 1)
    # for a real image |x * psi| = |x * conj(psi)|: each wavelet also covers the mirrored half of the plane
    wav = wav + np.roll(wav[::-1, ::-1], 1, axis=(0, 1))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8))
    for ax, image, title in [(axes[0], cnn, f"CNN: {len(kernels)} learned filters"),
                             (axes[1], wav, f"ScatNet: {J}x{L} Morlet wavelets")]:
        ax.imshow(image / image.max(), cmap=SEQUENTIAL, extent=[-0.5, 0.5, -0.5, 0.5])
        ax.set_title(title)
        ax.set_xlabel("horizontal frequency (cycles/pixel)")
        ax.set_ylabel("vertical frequency")
    fig.suptitle("Frequency coverage of the first layer (power spectrum, summed)")
    _finish(fig, save_to)


# ----------------------------------------------------------------------------- XAI

def _overlay(ax, image, attribution, smooth=1.0):
    """Grey X-ray + attribution (red = for the class, blue = against); weak values fade out."""
    ax.imshow(denormalize(image), cmap="gray", vmin=0, vmax=1)
    if np.isnan(attribution).all():
        ax.text(0.5, 0.5, "not\napplicable", transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color=INK, bbox=dict(facecolor=SURFACE, edgecolor=GRID, alpha=0.9))
    else:
        a = gaussian_filter(attribution, smooth) if smooth else attribution
        scale = np.percentile(np.abs(a), 99) or 1e-12
        a = np.clip(a / scale, -1, 1)
        ax.imshow(_rgba((a + 1) / 2, DIVERGING, 0.7 * np.abs(a)))
    ax.axis("off")


def plot_attribution_grid(images, labels, maps, class_names, methods=None, indices=None, title=None, save_to=None):
    """Rows = images, columns = the X-ray + one overlay per XAI method."""
    methods = list(methods or maps)
    indices = list(range(len(images))) if indices is None else list(indices)
    fig, axes = plt.subplots(len(indices), len(methods) + 1,
                             figsize=(2.0 * (len(methods) + 1), 2.15 * len(indices)), squeeze=False)
    for row, i in enumerate(indices):
        ax = axes[row, 0]
        ax.imshow(denormalize(images[i]), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"true: {class_names[labels[i]]}", fontsize=8.5, color=INK)
        for col, method in enumerate(methods, start=1):
            _overlay(axes[row, col], images[i], maps[method][i])
            if row == 0:
                axes[row, col].set_title(method, fontsize=9.5)
    if title:
        fig.suptitle(title, fontsize=12)
    _finish(fig, save_to)


def plot_scratch_vs_captum(images, attributions, index=0, save_to=None):
    """For one image: Captum's Occlusion, ours, and their absolute difference, one row per model."""
    fig, axes = plt.subplots(len(attributions), 4, figsize=(10, 2.6 * len(attributions)), squeeze=False)
    for row, (name, maps) in enumerate(attributions.items()):
        captum, ours = maps["Occlusion"][index], maps["Occlusion (ours)"][index]
        v = np.abs(captum).max() or 1e-12
        panels = [(denormalize(images[index]), "gray", 0, 1, f"{label(name)}: image"),
                  (captum, DIVERGING, -v, v, "Captum Occlusion"),
                  (ours, DIVERGING, -v, v, "Occlusion from scratch"),
                  (np.abs(captum - ours), SEQUENTIAL, 0, v, f"|difference| (max {np.abs(captum - ours).max():.1e})")]
        for ax, (img, cmap, vmin, vmax, text) in zip(axes[row], panels):
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(text, fontsize=9.5)
            ax.axis("off")
    _finish(fig, save_to)


def plot_agreement(matrices, save_to=None):
    """Method-agreement heatmaps (Spearman rho between the maps of two methods), one per model."""
    fig, axes = plt.subplots(1, len(matrices), figsize=(4.9 * len(matrices), 4.5), squeeze=False)
    for ax, (name, m) in zip(axes[0], matrices.items()):
        ax.imshow(m.values.astype(float), cmap=DIVERGING, vmin=-1, vmax=1)
        for i in range(len(m)):
            for j in range(len(m)):
                ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(m)), m.columns, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(m)), m.index, fontsize=8)
        ax.set_title(f"{label(name)}: agreement between methods")
        for spine in ax.spines.values():
            spine.set_visible(False)
    _finish(fig, save_to)


def plot_deletion_curves(curves, save_to=None):
    """Deletion test: probability of the class while the most important patches are removed (lower = better)."""
    fig, axes = plt.subplots(1, len(curves), figsize=(4.4 * len(curves), 3.6), squeeze=False, sharey=True)
    for ax, (name, by_method) in zip(axes[0], curves.items()):
        for method, c in by_method.items():
            x = np.linspace(0, 100, c.shape[1])
            auc = ((c[:, :-1] + c[:, 1:]) / 2).mean()
            ax.plot(x, c.mean(0), color=METHOD_COLORS.get(method, INK), label=f"{method} ({auc:.2f})",
                    linewidth=1.5 if method == "Random" else 2)
        ax.set_title(f"{label(name)}")
        ax.set_xlabel("% of the image removed (most important first)")
        _grid_axes(ax, "both")
        ax.legend(fontsize=7.5, title="method (area)", title_fontsize=8)
    axes[0, 0].set_ylabel("probability of the true class")
    fig.suptitle("Deletion test: lower = more faithful attribution")
    _finish(fig, save_to)


# ----------------------------------------------------------------------------- fracture boxes

def plot_fracture_boxes(images, maps, true_boxes, methods=None, threshold=0.5, title=None, save_to=None):
    """Rows = fractured X-rays, columns = methods: radiologist's box (dashed), our box, evidence (faint outside)."""
    methods = list(methods or maps)
    fig, axes = plt.subplots(len(images), len(methods), figsize=(2.0 * len(methods), 2.15 * len(images)),
                             squeeze=False)
    for row, (image, truth) in enumerate(zip(images, true_boxes)):
        for col, method in enumerate(methods):
            ax, attribution = axes[row, col], maps[method][row]
            ax.imshow(denormalize(image), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(method, fontsize=9.5)
            for box in truth:
                _draw_box(ax, box, "radiologist")
            if np.isnan(attribution).all():
                ax.text(0.5, 0.5, "not\napplicable", transform=ax.transAxes, ha="center", va="center",
                        fontsize=10, color=INK, bbox=dict(facecolor=SURFACE, edgecolor=GRID, alpha=0.9))
                continue
            heat = evidence(attribution)
            heat = heat / (heat.max() or 1)
            alpha = 0.25 * heat
            box = heatmap_to_box(attribution, threshold)
            if box is not None:
                x0, y0, x1, y1 = box.astype(int)
                alpha[y0:y1, x0:x1] = 0.7 * heat[y0:y1, x0:x1]
                _draw_box(ax, box, "ours")
            ax.imshow(_rgba(heat, POSITIVE, alpha))
            hit = "hit" if pointing_game(attribution, truth) else "miss"
            ax.text(0.03, 0.03, f"{hit}, IoU {iou(box, truth):.2f}", transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=7.5, color=INK, bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=1.5))
    handles = [Line2D([], [], color=BOX_COLORS["radiologist"], linestyle="--", label="radiologist's box"),
               Line2D([], [], color=BOX_COLORS["ours"], label="our box (from the map)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    if title:
        fig.suptitle(title, fontsize=12)
    _finish(fig, save_to)


def plot_localization(tables, detector=None, save_to=None):
    """Pointing game per method and model, with a random point and YOLO as dashed lines."""
    methods = [m for m in METHOD_COLORS if m not in ("Random", "Occlusion (ours)")
               and any(m in t.index for t in tables.values())]
    random = next(iter(tables.values())).loc["Random", "hit rate (%)"]  # same images for every model
    width = 0.8 / len(tables)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for k, (name, t) in enumerate(tables.items()):
        x = np.arange(len(methods)) + (k - (len(tables) - 1) / 2) * width
        values = [t.loc[m, "hit rate (%)"] if m in t.index else np.nan for m in methods]
        ax.bar(x, values, width * 0.92, color=_color(name), label=label(name))
        for xi, v in zip(x, values):
            if np.isnan(v):
                ax.text(xi, random + 2, "n/a", ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.axhline(random, color=NEUTRAL, linestyle="--", linewidth=1.2, label=f"random point ({random:.0f}%)")
    if detector is not None:
        ax.axhline(detector, color=BOX_COLORS["detector"], linestyle="--", linewidth=1.5,
                   label=f"YOLO, trained on the boxes ({detector:.0f}%)")
    ax.set_xticks(range(len(methods)), [m.replace(" ", "\n") for m in methods])
    ax.set_ylim(0, 100)
    ax.set_ylabel("hottest point inside\nthe radiologist's box (%)")
    ax.set_title("Does the explanation point at the fracture? (fractured test X-rays)")
    _grid_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(tables) + 1 + (detector is not None))
    _finish(fig, save_to)


def plot_pipeline_examples(image_set, cases, n=4, save_to=None):
    """A few test X-rays through the pipeline: radiologist's, YOLO's and the explanation's box + the report."""
    kinds = [(cases["true"] == "fractured") & (cases["verdict"] == "fracture"),
             cases["verdict"] == "needs review",
             (cases["true"] != "fractured") & (cases["verdict"] == "no fracture"),
             ((cases["true"] == "fractured") & (cases["verdict"] == "no fracture"))
             | ((cases["true"] != "fractured") & (cases["verdict"] == "fracture"))]
    picked = [int(np.flatnonzero(k)[0]) for k in kinds if k.any()]
    picked += [i for i in range(len(cases)) if i not in picked][:max(0, n - len(picked))]
    fig, axes = plt.subplots(1, len(picked[:n]), figsize=(3.3 * len(picked[:n]), 4.9), squeeze=False)
    for ax, i in zip(axes[0], picked[:n]):
        ax.imshow(image_set.images[i], cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        for box in image_set.boxes[i]:
            _draw_box(ax, box, "radiologist")
        for key, who in (("detector box", "detector"), ("xai box", "ours")):
            if isinstance(cases[key].iloc[i], np.ndarray):
                _draw_box(ax, cases[key].iloc[i], who)
        v, fractured = cases["verdict"].iloc[i], cases["true"].iloc[i] == "fractured"
        outcome = ("" if v == "needs review" else " (correct)" if (v == "fracture") == fractured
                   else " (missed fracture)" if fractured else " (false alarm)")
        ax.set_title(v + outcome, fontsize=11, color=INK if outcome == " (correct)" else "#d03b3b")
        lines = cases["report"].iloc[i].split("\n")[1:]  # without the file-name line
        ax.text(0.0, -0.03, "\n".join(textwrap.fill(line, 46, subsequent_indent="  ") for line in lines),
                transform=ax.transAxes, ha="left", va="top", fontsize=7, color=INK_2)
    handles = [Line2D([], [], color=BOX_COLORS["radiologist"], linestyle="--", label="radiologist's box"),
               Line2D([], [], color=BOX_COLORS["detector"], label="YOLO's box"),
               Line2D([], [], color=BOX_COLORS["ours"], label="explanation's box")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    _finish(fig, save_to)
