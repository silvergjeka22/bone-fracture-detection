"""From an XAI heatmap to a fracture box, scored against the radiologists' boxes.

hit rate = the hottest point is inside a radiologist's box (pointing game); IoU = box overlap.
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import binomtest

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


def _inside(x, y, boxes):
    """Is the point (x, y) inside one of `boxes` (k, >= 4)?"""
    boxes = np.asarray(boxes, dtype=float)
    if boxes.size == 0:
        return False
    x0, y0, x1, y1 = boxes.reshape(-1, boxes.shape[-1])[:, :4].T
    return bool(((x0 <= x) & (x <= x1) & (y0 <= y) & (y <= y1)).any())


def pointing_game(attribution, true_boxes, smooth=SMOOTH):
    """Does the hottest point of the (positive, smoothed) map lie inside one of `true_boxes`?"""
    heat = evidence(attribution, smooth)
    if heat.max() <= 0 or len(true_boxes) == 0:
        return False
    row, col = np.unravel_index(heat.argmax(), heat.shape)
    return _inside(col + 0.5, row + 0.5, true_boxes)  # centre of the pixel, in the coordinates of the boxes


def box_area_share(true_boxes, shape):
    """Share of the image covered by `true_boxes`: the hit rate of a random point."""
    covered = np.zeros(shape, bool)
    for x0, y0, x1, y1 in np.asarray(true_boxes, dtype=float).reshape(-1, 4):
        covered[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = True
    return covered.mean()


def localization_table(maps, true_boxes, threshold=THRESHOLD, smooth=SMOOTH):
    """{method: (N, H, W) maps} of N fractured X-rays -> hit rate, mean IoU and mean box size per method."""
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


def box_hits(boxes, true_boxes):
    """Per X-ray: is the centre of its first (most confident) box inside one of its true boxes?"""
    return np.array([len(b) > 0 and _inside((b[0, 0] + b[0, 2]) / 2, (b[0, 1] + b[0, 3]) / 2, t)
                     for b, t in zip(boxes, true_boxes)], dtype=bool)


def mcnemar_hits(a, b):
    """Exact McNemar test on paired hits of the same X-rays: only a, only b, p-value (is the gain real?)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    only_a, only_b = int((a & ~b).sum()), int((b & ~a).sum())
    p = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return only_a, only_b, round(float(p), 4)


def score_boxes(boxes, true_boxes, shape):
    """Hit rate, IoU and box size of predicted boxes (the most confident box of each image)."""
    top = [b[0, :4] if len(b) else None for b in boxes]
    area = shape[0] * shape[1]
    return {"hit rate (%)": round(100 * float(np.mean(box_hits(boxes, true_boxes))), 1),
            "IoU": round(float(np.mean([iou(b, t) for b, t in zip(top, true_boxes)])), 3),
            "box size (%)": round(100 * float(np.mean([0.0 if b is None else (b[2] - b[0]) * (b[3] - b[1]) / area
                                                        for b in top])), 1)}


def detector_vs_xai(localization, detector_row, model_labels=None):
    """One row per box source: the detector, the best XAI method of every model, a random point."""
    model_labels = model_labels or {}
    rows = {"YOLO (trained on boxes)": detector_row}
    for name, table in localization.items():
        methods = table.drop("Random")
        rows[f"{model_labels.get(name, name)}: {methods.index[0]}"] = methods.iloc[0].to_dict()
    rows["Random point"] = next(iter(localization.values())).loc["Random"].to_dict()
    return pd.DataFrame(rows).T


def average_precision(found, true_boxes, threshold=0.5):
    """AP at IoU >= threshold (as in mAP@0.5): scored boxes (k, 5) of every X-ray against its true boxes."""
    n_true, scored = sum(len(t) for t in true_boxes), []
    for boxes, truth in zip(found, true_boxes):
        truth, used = np.asarray(truth, dtype=float).reshape(-1, 4), np.zeros(len(truth), bool)
        for box in boxes[np.argsort(-boxes[:, 4], kind="stable")]:
            overlaps = np.array([iou(box[:4], t) for t in truth])
            j = int(overlaps.argmax()) if len(truth) else -1
            hit = j >= 0 and overlaps[j] >= threshold and not used[j]
            if hit:
                used[j] = True  # each true box is found once; a second box on it is a false positive
            scored.append((box[4], hit))
    if not scored or n_true == 0:
        return 0.0
    hits = np.array([h for _, h in sorted(scored, key=lambda s: -s[0])])
    recall = np.cumsum(hits) / n_true
    precision = np.maximum.accumulate((np.cumsum(hits) / np.arange(1, len(hits) + 1))[::-1])[::-1]
    return float(np.sum(np.diff(recall, prepend=0) * precision))


def agreement(maps, found, smooth=SMOOTH):
    """Share (%) of X-rays whose map has its hottest point inside the detector's most confident box (no box: no)."""
    return 100 * float(np.mean([len(b) > 0 and pointing_game(a, b[:1, :4], smooth) for a, b in zip(maps, found)]))


def compare_hits(maps_a, maps_b, true_boxes, names=("A", "C"), smooth=SMOOTH):
    """Per method: hits of two models on the same X-rays and the exact McNemar p-value (is the gain real?)."""
    rows = {}
    for method in maps_a:
        if method not in maps_b or np.isnan(maps_a[method]).all():
            continue
        a = np.array([pointing_game(m, t, smooth) for m, t in zip(maps_a[method], true_boxes)])
        b = np.array([pointing_game(m, t, smooth) for m, t in zip(maps_b[method], true_boxes)])
        only_a, only_b, p = mcnemar_hits(a, b)
        rows[method] = {f"hits {names[0]}": int(a.sum()), f"hits {names[1]}": int(b.sum()),
                        f"only {names[0]}": only_a, f"only {names[1]}": only_b, "p-value": p}
    return pd.DataFrame(rows).T
