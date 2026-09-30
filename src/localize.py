"""From an XAI heatmap to a fracture box, scored against the boxes drawn by the radiologists.

A heatmap colours the whole X-ray; a box says "the fracture is here", which anyone can check:

    1. keep only the evidence FOR 'fractured' (the positive values) and smooth it a little
       (gradient maps are noisy pixel by pixel);
    2. keep the pixels above `threshold` x the strongest evidence;
    3. the connected blob that holds the most evidence is our fracture: the box goes around it.

Scores over the fractured test X-rays (FracAtlas has a box around every fracture):
    hit rate      pointing game: the hottest point of the map lies inside a radiologist's box
    IoU           overlap (intersection / union) of our box with the best-matching radiologist's box
    box size      area of our box, % of the image (a huge box would contain the fracture by luck)
'Random' is the reference to beat: the chance that a random point lies inside a radiologist's box.
"""

import numpy as np
import pandas as pd
from scipy import ndimage

THRESHOLD = 0.5  # share of the strongest evidence a pixel needs to be part of the box
SMOOTH = 4       # Gaussian sigma in pixels


def evidence(attribution, smooth=SMOOTH):
    """Positive part of a map (evidence FOR the class), smoothed; a NaN map (method not applicable) -> zeros."""
    heat = np.clip(np.nan_to_num(np.asarray(attribution, dtype=float)), 0, None)
    return ndimage.gaussian_filter(heat, smooth) if smooth else heat


def heatmap_to_box(attribution, threshold=THRESHOLD, smooth=SMOOTH):
    """Our box (x0, y0, x1, y1) in pixels around the strongest evidence, or None if there is no evidence."""
    heat = evidence(attribution, smooth)
    if heat.max() <= 0:
        return None
    blobs, n = ndimage.label(heat >= threshold * heat.max())
    mass = ndimage.sum(heat, blobs, index=np.arange(1, n + 1))
    rows, cols = np.nonzero(blobs == np.argmax(mass) + 1)
    return np.array([cols.min(), rows.min(), cols.max() + 1, rows.max() + 1], dtype=float)


def iou(box, true_boxes):
    """Best intersection over union of `box` with one of `true_boxes` (k, 4); 0 without a box."""
    true_boxes = np.asarray(true_boxes, dtype=float).reshape(-1, 4)
    if box is None or len(true_boxes) == 0:
        return 0.0
    x0, y0, x1, y1 = true_boxes.T
    inter = (np.clip(np.minimum(box[2], x1) - np.maximum(box[0], x0), 0, None)
             * np.clip(np.minimum(box[3], y1) - np.maximum(box[1], y0), 0, None))
    union = (box[2] - box[0]) * (box[3] - box[1]) + (x1 - x0) * (y1 - y0) - inter
    return float((inter / union).max())


def pointing_game(attribution, true_boxes, smooth=SMOOTH):
    """Does the hottest point of the (positive, smoothed) map lie inside one of `true_boxes`?"""
    heat = evidence(attribution, smooth)
    if heat.max() <= 0:
        return False
    row, col = np.unravel_index(heat.argmax(), heat.shape)
    x, y = col + 0.5, row + 0.5  # centre of the pixel, in the coordinates of the boxes
    x0, y0, x1, y1 = np.asarray(true_boxes, dtype=float).reshape(-1, 4).T
    return bool(((x0 <= x) & (x <= x1) & (y0 <= y) & (y <= y1)).any())


def box_area_share(true_boxes, shape):
    """Share of the image covered by `true_boxes`: the hit rate of a random point."""
    covered = np.zeros(shape, bool)
    for x0, y0, x1, y1 in np.asarray(true_boxes, dtype=float).reshape(-1, 4):
        covered[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = True
    return covered.mean()


def localization_table(maps, true_boxes, threshold=THRESHOLD, smooth=SMOOTH):
    """{method: (N, H, W) maps} of N fractured X-rays -> hit rate, mean IoU and mean box size per method.

    Methods that do not apply to the model (NaN maps) are left out; 'Random' is the reference.
    """
    rows = {}
    for method, m in maps.items():
        if np.isnan(m).all():
            continue
        boxes = [heatmap_to_box(a, threshold, smooth) for a in m]
        rows[method] = {
            "hit rate (%)": 100 * np.mean([pointing_game(a, t, smooth) for a, t in zip(m, true_boxes)]),
            "IoU": np.mean([iou(b, t) for b, t in zip(boxes, true_boxes)]),
            "box size (%)": 100 * np.mean([0.0 if b is None else (b[2] - b[0]) * (b[3] - b[1]) / a.size
                                           for a, b in zip(m, boxes)]),
        }
    shape = next(iter(maps.values())).shape[1:]
    rows["Random"] = {"hit rate (%)": 100 * np.mean([box_area_share(t, shape) for t in true_boxes]),
                      "IoU": np.nan, "box size (%)": np.nan}
    table = pd.DataFrame(rows).T.sort_values("hit rate (%)", ascending=False)
    return table.round({"hit rate (%)": 1, "IoU": 3, "box size (%)": 1})
