"""Six XAI methods (Captum) applied to every model, our Occlusion, and how to compare attributions.

    method                family              what it needs from the model        ScatNet
    Saliency              gradient            gradients                           yes
    Integrated Gradients  gradient (path)     gradients + a baseline image        yes
    Guided Backprop       modified gradient   ReLU layers                         only in the classifier
    Grad-CAM              class activation    a LEARNED convolutional feature map NO (fixed wavelets)
    Occlusion             perturbation        forward passes only                 yes
    LIME                  local surrogate     forward passes only                 yes

Every method returns one (H, W) map for the target class, larger = more evidence for that class:
Saliency and Guided Backprop give gradient magnitudes, Grad-CAM is >= 0, Integrated Gradients,
Occlusion and LIME are signed (negative = evidence against the class). A method that does not
apply to a model returns None (stored as a NaN map).

"Removed" pixels (baselines of Integrated Gradients, Occlusion, LIME and of the deletion test)
are set to black, the X-ray background: a grey patch on a black background would be an
unrealistic image and move the score for the wrong reason.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum._utils.models.linear_model import SkLearnRidge
from captum.attr import (GuidedBackprop, IntegratedGradients, LayerAttribution, LayerGradCam, Lime, Occlusion,
                         Saliency)
from scipy.stats import spearmanr

from .data import BLACK
from .occlusion_scratch import compare, occlusion_scratch

# Captum warns on every call that it enables input gradients / hooks the ReLUs: expected, not a problem
warnings.filterwarnings("ignore", message=".*did not already require gradients.*")
warnings.filterwarnings("ignore", message=".*Setting backward hooks on ReLU activations.*")

DEFAULT_PARAMS = dict(baseline=BLACK, ig_steps=64, window=16, stride=8, lime_samples=1000, patch=16)


def _to_map(attribution):
    """(1, C, H, W) attribution -> (H, W) numpy map (sum over channels)."""
    return attribution.detach()[0].sum(0).cpu().numpy()


def _device(model):
    return next(model.parameters(), torch.empty(0)).device


def patch_mask(x, patch=16):
    """LIME features: a grid of patch x patch squares (14 x 14 = 196 regions for 224 px)."""
    _, _, height, width = x.shape
    ids = torch.arange((height // patch) * (width // patch)).reshape(height // patch, width // patch)
    return ids.repeat_interleave(patch, 0).repeat_interleave(patch, 1)[None, None].to(x.device)


# Every method: (model, x of shape (1, 1, H, W), target class, **params) -> (H, W) map or None.

def saliency(model, x, target, **_):
    return _to_map(Saliency(model).attribute(x, target=target, abs=True))


def integrated_gradients(model, x, target, baseline=BLACK, ig_steps=64, **_):
    ig = IntegratedGradients(model)
    return _to_map(ig.attribute(x, baselines=baseline, target=target, n_steps=ig_steps,
                                internal_batch_size=min(ig_steps, 32)))


def guided_backprop(model, x, target, **_):
    return np.abs(_to_map(GuidedBackprop(model).attribute(x, target=target)))


def grad_cam(model, x, target, **_):
    layer = getattr(model, "cam_layer", None)
    if layer is None:  # ScatNet: its only feature maps are fixed wavelet coefficients, nothing learned
        return None
    cam = LayerGradCam(model, layer).attribute(x, target=target, relu_attributions=True)
    return _to_map(LayerAttribution.interpolate(cam, tuple(x.shape[-2:]), interpolate_mode="bilinear"))


def occlusion(model, x, target, baseline=BLACK, window=16, stride=8, **_):
    return _to_map(Occlusion(model).attribute(x, target=target, baselines=baseline,
                                              sliding_window_shapes=(x.shape[1], window, window),
                                              strides=(x.shape[1], stride, stride), perturbations_per_eval=64))


def occlusion_ours(model, x, target, baseline=BLACK, window=16, stride=8, **_):
    return occlusion_scratch(model, x, target, window=window, stride=stride, baseline=baseline)


def lime(model, x, target, baseline=BLACK, lime_samples=1000, patch=16, **_):
    # ridge surrogate as in the original LIME (Captum's default Lasso often zeroes every patch)
    surrogate = SkLearnRidge(alpha=1.0)
    return _to_map(Lime(model, interpretable_model=surrogate).attribute(x, target=target, baselines=baseline, feature_mask=patch_mask(x, patch),
                                         n_samples=lime_samples, perturbations_per_eval=32))


METHODS = {
    "Saliency": saliency,
    "Integrated Gradients": integrated_gradients,
    "Guided Backprop": guided_backprop,
    "Grad-CAM": grad_cam,
    "Occlusion": occlusion,
    "LIME": lime,
}
SCRATCH = {"Occlusion (ours)": occlusion_ours}


def applicability(models):
    """Can each method be used on each model? ({name: model} -> table)."""
    rows = {}
    for name, model in models.items():
        has_cam = getattr(model, "cam_layer", None) is not None
        relu_in_features = any(isinstance(m, nn.ReLU) for m in model.modules()) and has_cam
        rows[name] = {method: "yes" for method in METHODS}
        rows[name]["Guided Backprop"] = "yes" if relu_in_features else "only classifier ReLUs"
        rows[name]["Grad-CAM"] = "yes" if has_cam else "no: no learned conv layer"
    return pd.DataFrame(rows)


def explain_all(model, images, targets, methods=None, cache=None, recompute=False, **params):
    """Maps of every method for every image: ({method: (N, H, W)}, {method: seconds per image}).

    If `cache` (.npz) exists the maps are loaded instead of recomputed (unless recompute=True).
    """
    methods = METHODS if methods is None else methods
    params = {**DEFAULT_PARAMS, **params}
    if cache is not None and Path(cache).exists() and not recompute:
        with np.load(cache) as f:
            maps = {k: f[k] for k in f.files if k != "_seconds"}
            seconds = dict(zip(maps, f["_seconds"].tolist()))
        return maps, seconds

    device = _device(model)
    model.eval()
    maps, seconds = {}, {}
    for name, method in methods.items():
        start = time.time()
        out = [method(model, image[None].to(device), int(target), **params) for image, target in zip(images, targets)]
        seconds[name] = (time.time() - start) / len(images)
        maps[name] = (np.full((len(images),) + tuple(images.shape[-2:]), np.nan, np.float32) if out[0] is None
                      else np.stack(out).astype(np.float32))
        print(f"  {name:22s} {seconds[name]:6.2f} s/image" + ("  (not applicable)" if out[0] is None else ""))

    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **maps, _seconds=np.array([seconds[k] for k in maps]))
    return maps, seconds


def available(maps):
    """Only the methods that could be computed (no NaN maps)."""
    return {k: v for k, v in maps.items() if not np.isnan(v).all()}


# ----------------------------------------------------------------------------- comparing attributions

@torch.no_grad()
def deletion_curve(model, x, target, attribution, patch=16, steps=20, baseline=BLACK):
    """Probability of the target class while the patches are blacked out, most important first.

    A faithful map puts the patches the model really uses first, so the probability drops fast:
    the smaller the area under this curve, the better the attribution.
    """
    _, _, height, width = x.shape
    gh, gw = height // patch, width // patch
    scores = attribution[:gh * patch, :gw * patch].reshape(gh, patch, gw, patch).mean((1, 3)).ravel()
    rank = np.empty(len(scores), int)
    rank[np.argsort(-scores, kind="stable")] = np.arange(len(scores))
    counts = np.linspace(0, len(scores), steps + 1).round().astype(int)
    removed = torch.from_numpy(rank[None, :] < counts[:, None]).view(-1, 1, gh, gw)
    removed = removed.repeat_interleave(patch, 2).repeat_interleave(patch, 3)
    removed = nn.functional.pad(removed, (0, width - gw * patch, 0, height - gh * patch)).to(x.device)
    batch = torch.where(removed, torch.full_like(x, baseline), x.expand(len(counts), -1, -1, -1))
    return model(batch).softmax(1)[:, target].cpu().numpy()


def deletion_curves(model, images, targets, maps, patch=16, steps=20, n_random=5, seed=0):
    """{method: (N, steps + 1) probabilities}, plus 'Random' (random patch order, the reference to beat)."""
    device = _device(model)
    model.eval()
    rng = np.random.default_rng(seed)
    curves = {}
    for name, m in available(maps).items():
        if name not in METHODS:  # our Occlusion equals Captum's: compared on its own in scratch_vs_captum
            continue
        curves[name] = np.stack([deletion_curve(model, img[None].to(device), int(t), a, patch, steps)
                                 for img, t, a in zip(images, targets, m)])
    height, width = images.shape[-2:]
    curves["Random"] = np.stack([
        np.mean([deletion_curve(model, img[None].to(device), int(t), rng.random((height, width)), patch, steps)
                 for _ in range(n_random)], axis=0)
        for img, t in zip(images, targets)])
    return curves


def deletion_table(curves):
    """Area under the deletion curve per method (mean ± std over the images, lower = more faithful)."""
    auc = {name: ((c[:, :-1] + c[:, 1:]) / 2).mean(1) for name, c in curves.items()}
    table = pd.DataFrame({"deletion AUC (mean)": {n: a.mean() for n, a in auc.items()},
                          "std": {n: a.std() for n, a in auc.items()}})
    return table.sort_values("deletion AUC (mean)").round(4)


def agreement_matrix(maps, patch=16):
    """Spearman rank correlation between methods, averaged over images.

    Maps are compared as |attribution| averaged over patch x patch regions, so that pixel-level
    methods (gradients) and region-level methods (LIME, Occlusion, Grad-CAM) are on equal footing.
    """
    def pooled(a):  # (N, H, W) -> (N, regions)
        n, h, w = a.shape
        a = a[:, :h // patch * patch, :w // patch * patch]
        return np.abs(a).reshape(n, h // patch, patch, w // patch, patch).mean(axis=(2, 4)).reshape(n, -1)

    maps = {k: v for k, v in available(maps).items() if k in METHODS}
    names = list(maps)
    regions = {name: pooled(maps[name]) for name in names}
    matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rho = np.nanmean([spearmanr(u, v).statistic for u, v in zip(regions[a], regions[b])])
            matrix.loc[a, b] = matrix.loc[b, a] = rho
    return matrix


def scratch_vs_captum(attributions):
    """How close our Occlusion is to Captum's, one row per model."""
    rows = {}
    for model_name, maps in attributions.items():
        table = compare(maps["Occlusion (ours)"], maps["Occlusion"])
        rows[model_name] = {"mean Pearson r": table["pearson r"].mean(),
                            "min Pearson r": table["pearson r"].min(),
                            "max |difference|": table["max |difference|"].max(),
                            "max |attribution|": table["max |captum|"].max()}
    return pd.DataFrame(rows).T
