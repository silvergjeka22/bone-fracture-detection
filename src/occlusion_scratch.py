"""Occlusion implemented from scratch (Zeiler & Fergus 2014), compared with Captum's.

Slide a black window over the image; the drop of the class score is the importance of the covered
pixels; each pixel gets the mean drop of the windows that cover it.
"""

import math

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def occlusion_scratch(model, x, target, window=16, stride=8, baseline=-1.0, batch_size=64):
    """Occlusion map (H, W) for one image `x` of shape (1, C, H, W)."""
    model.eval()
    _, _, height, width = x.shape
    base_score = model(x)[0, target]

    # top-left corners of the windows; the last window may be clipped by the border
    rows = [i * stride for i in range(math.ceil((height - window) / stride) + 1)]
    cols = [j * stride for j in range(math.ceil((width - window) / stride) + 1)]
    corners = [(r, c) for r in rows for c in cols]

    total = torch.zeros(height, width, device=x.device)  # sum of score drops per pixel
    count = torch.zeros(height, width, device=x.device)  # number of windows covering each pixel
    for start in range(0, len(corners), batch_size):     # occluded copies are scored in batches
        chunk = corners[start:start + batch_size]
        batch = x.repeat(len(chunk), 1, 1, 1)
        for k, (r, c) in enumerate(chunk):
            batch[k, :, r:r + window, c:c + window] = baseline
        drops = base_score - model(batch)[:, target]
        for k, (r, c) in enumerate(chunk):
            total[r:r + window, c:c + window] += drops[k]
            count[r:r + window, c:c + window] += 1
    return (total / count.clamp_min(1)).cpu().numpy()


def compare(scratch_maps, captum_maps):
    """Per-image agreement between our maps and Captum's: Pearson r and largest absolute difference."""
    rows = [{"pearson r": np.corrcoef(s.ravel(), c.ravel())[0, 1],
             "max |difference|": np.abs(s - c).max(),
             "max |captum|": np.abs(c).max()}
            for s, c in zip(scratch_maps, captum_maps)]
    return pd.DataFrame(rows)
