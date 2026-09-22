"""Six XAI methods (Captum) applied in the same way to every model, plus caching and comparison.

Each method returns one (H, W) map per image: the contribution of every pixel to the score of
the `target` class, averaged over the 3 (identical, grey) input channels.
Positive = evidence for the target class, negative = evidence against it
(Saliency is the absolute gradient, so it is always >= 0).

All methods are model-agnostic (they only need gradients or forward passes), which is why
they can be applied to the CNN, ResNet18 and ScatNet alike. Layer-based methods such as
GradCAM are not used: ScatNet has no learned convolutional feature maps to hook into.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from captum.attr import InputXGradient, IntegratedGradients, Lime, Occlusion, Saliency, ShapleyValueSampling
from scipy.stats import spearmanr

from .occlusion_scratch import compare, occlusion_scratch


def _to_map(attribution):
    """(1, C, H, W) attribution -> (H, W) numpy map."""
    return attribution.detach().squeeze(0).mean(0).cpu().numpy()


def patch_mask(x, patch=16):
    """LIME / Shapley features: a grid of patch x patch squares (14 x 14 = 196 regions for 224 px)."""
    _, _, height, width = x.shape
    ids = torch.arange((height // patch) * (width // patch)).reshape(height // patch, width // patch)
    return ids.repeat_interleave(patch, 0).repeat_interleave(patch, 1)[None, None].to(x.device)


# Every method takes (model, x of shape (1, C, H, W), target class) and returns an (H, W) map.
# Extra keyword arguments are ignored, so one parameter dict can be passed to all of them.

def saliency(model, x, target, **_):
    return _to_map(Saliency(model).attribute(x, target=target))


def input_x_gradient(model, x, target, **_):
    return _to_map(InputXGradient(model).attribute(x, target=target))


def integrated_gradients(model, x, target, baseline=0.0, ig_steps=100, **_):
    ig = IntegratedGradients(model)
    return _to_map(ig.attribute(x, baselines=baseline, target=target, n_steps=ig_steps))


def lime(model, x, target, baseline=0.0, lime_samples=1000, **_):
    return _to_map(Lime(model).attribute(x, target=target, baselines=baseline,
                                         feature_mask=patch_mask(x), n_samples=lime_samples))


def shapley(model, x, target, baseline=0.0, shapley_samples=50, **_):
    svs = ShapleyValueSampling(model)
    return _to_map(svs.attribute(x, target=target, baselines=baseline, feature_mask=patch_mask(x),
                                 n_samples=shapley_samples, perturbations_per_eval=10))


def occlusion(model, x, target, baseline=0.0, occlusion_window=8, occlusion_stride=4, **_):
    window = (x.shape[1], occlusion_window, occlusion_window)  # occlude all channels at once
    return _to_map(Occlusion(model).attribute(x, target=target, baselines=baseline,
                                              sliding_window_shapes=window, strides=occlusion_stride))


def occlusion_ours(model, x, target, baseline=0.0, occlusion_window=8, occlusion_stride=4, **_):
    return occlusion_scratch(model, x, target, window=occlusion_window, stride=occlusion_stride,
                             baseline=baseline)


CAPTUM_METHODS = {
    "Saliency": saliency,
    "Input x Gradient": input_x_gradient,
    "Integrated Gradients": integrated_gradients,
    "LIME": lime,
    "Shapley": shapley,
    "Occlusion": occlusion,
}
METHODS = {**CAPTUM_METHODS, "Occlusion (scratch)": occlusion_ours}


def explain_all(model, images, targets, methods=METHODS, cache=None, **params):
    """Attribution maps of every method for every image: {method: array (N, H, W)}.

    If `cache` is an existing .npz file the maps are loaded from it instead of recomputed
    (delete the file to recompute, e.g. after changing the parameters).
    """
    if cache is not None and Path(cache).exists():
        print(f"loaded {cache}")
        return load_attributions(cache)

    device = next(model.parameters()).device
    model.eval()
    maps = {}
    for method_name, method in methods.items():
        maps[method_name] = np.stack([method(model, image[None].to(device), int(target), **params)
                                      for image, target in zip(images, targets)])
        print(f"  {method_name:22s} done")

    if cache is not None:
        save_attributions(maps, cache)
    return maps


def save_attributions(maps, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **maps)


def load_attributions(path):
    with np.load(path) as f:
        return {name: f[name] for name in f.files}


def scratch_vs_captum(attributions):
    """How close our Occlusion is to Captum's, one row per model."""
    rows = {}
    for model_name, maps in attributions.items():
        table = compare(maps["Occlusion (scratch)"], maps["Occlusion"])
        rows[model_name] = {"mean pearson r": table["pearson r"].mean(),
                            "min pearson r": table["pearson r"].min(),
                            "max |difference|": table["max |difference|"].max(),
                            "max |attribution|": table["max |captum|"].max()}
    return pd.DataFrame(rows).T


def agreement_matrix(maps, patch=16):
    """Spearman rank correlation between methods, averaged over images.

    Maps are compared as |attribution| averaged over patch x patch regions, so that pixel-level
    methods (gradients) and region-level methods (LIME, Shapley, Occlusion) are on equal footing.
    """
    def pooled(a):  # (N, H, W) -> (N, regions)
        n, h, w = a.shape
        return np.abs(a).reshape(n, h // patch, patch, w // patch, patch).mean(axis=(2, 4)).reshape(n, -1)

    names = list(maps)
    regions = {name: pooled(maps[name]) for name in names}
    matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rho = np.nanmean([spearmanr(u, v).correlation for u, v in zip(regions[a], regions[b])])
            matrix.loc[a, b] = matrix.loc[b, a] = rho
    return matrix
