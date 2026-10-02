"""Full pipeline after Linda (2025): classifier + YOLO box + explanation -> verdict and report per X-ray.

YOLO proposes, the classifier decides: YOLO's candidate boxes are re-ranked with the classifier's Grad-CAM (fuse).
Verdict: 'fracture' / 'no fracture' when classifier and detector agree, otherwise 'needs review'.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .localize import (average_precision, box_hits, evidence, heatmap_to_box, localization_table, mcnemar_hits,
                       pointing_game, score_boxes)
from .xai import DEFAULT_PARAMS, METHODS, explain_all

VERDICTS = ("fracture", "no fracture", "needs review")


def verdict(p_fractured, has_box, threshold=0.5):
    """'fracture' / 'no fracture' when classifier and detector agree, 'needs review' when they do not."""
    if p_fractured >= threshold and has_box:
        return "fracture"
    if p_fractured < threshold and not has_box:
        return "no fracture"
    return "needs review"


def box_heat(cam, boxes):
    """Strongest evidence of the map inside each box, as a share of its strongest evidence anywhere (0 to 1)."""
    heat = evidence(cam)
    heat = heat / (heat.max() or 1)
    rows, cols = heat.shape
    out = []
    for x0, y0, x1, y1 in np.asarray(boxes, dtype=float).reshape(-1, 4):
        c0, r0 = int(np.clip(x0, 0, cols - 1)), int(np.clip(y0, 0, rows - 1))
        out.append(heat[r0:max(int(np.ceil(y1)), r0 + 1), c0:max(int(np.ceil(x1)), c0 + 1)].max())
    return np.array(out)


def fuse(detections, cams, p_fractured, alpha=0.5, threshold=None, top=10):
    """YOLO proposes, the classifier decides: score = p(fractured) x conf^(1 - alpha) x heat^alpha, best box first.

    heat = the classifier's Grad-CAM inside the box (box_heat); with `threshold`, X-rays called healthy lose their boxes.
    """
    fused = []
    for found, cam, p in zip(detections, cams, p_fractured):
        found = found[np.argsort(-found[:, 4], kind="stable")][:top]  # YOLO's candidates
        if len(found) == 0 or (threshold is not None and p < threshold):
            fused.append(np.zeros((0, 5), np.float32))
            continue
        score = p * found[:, 4] ** (1 - alpha) * box_heat(cam, found[:, :4]) ** alpha
        order = np.argsort(-score, kind="stable")
        fused.append(np.column_stack([found[order, :4], score[order]]).astype(np.float32))
    return fused


def choose_alpha(detections, cams, p_fractured, true_boxes, grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Weight of the Grad-CAM with the most hits on fractured (val) X-rays; 0 = YOLO's own order wins ties."""
    hits = {a: 100 * float(box_hits(fuse(detections, cams, p_fractured, a), true_boxes).mean()) for a in grid}
    return max(grid, key=lambda a: (hits[a], -a)), hits


def fusion_table(candidates, true_boxes, size):
    """Per box finder: hit rate and IoU on the fractured X-rays, McNemar vs the first row, AP@0.5 (fractured / all)."""
    frac = [i for i, t in enumerate(true_boxes) if len(t)]
    truth, rows, first = [true_boxes[i] for i in frac], {}, None
    for name, found in candidates.items():
        on_frac = [found[i] for i in frac]
        hits = box_hits(on_frac, truth)
        first = hits if first is None else first
        rows[name] = {**score_boxes(on_frac, truth, (size, size)), "McNemar p vs the first row": mcnemar_hits(first, hits)[2],
                      "AP@0.5, fractured X-rays (%)": round(100 * average_precision(on_frac, truth), 1),
                      "AP@0.5, all X-rays (%)": round(100 * average_precision(found, true_boxes), 1)}
    return pd.DataFrame(rows).T


def pick_method(model, val_set, threshold=0.5, **params):
    """XAI method with the most hits on the fractured val X-rays (the test set is never used to choose)."""
    idx = np.flatnonzero(val_set.labels == 0)
    images, _ = val_set.tensors(idx)
    maps, _ = explain_all(model, images, np.zeros(len(idx), dtype=int), **params)
    table = localization_table(maps, [val_set.boxes[i] for i in idx], threshold)
    return table.drop("Random").index[0], table


def run(model, image_set, probs, detections, method, device, threshold=0.5, **params):
    """One row per X-ray of `image_set`: classifier probability, detector box, explanation box, verdict, report."""
    params = {**DEFAULT_PARAMS, **params}
    model.eval()
    rows = []
    for i, (p, found) in enumerate(zip(np.asarray(probs, dtype=float), detections)):
        box = found[0, :4] if len(found) else None
        xai_box, agrees = None, None
        if p >= threshold:  # explain the positive findings only
            x, _ = image_set.tensors([i])
            attribution = METHODS[method](model, x.to(device), 0, **params)
            xai_box = heatmap_to_box(attribution)
            agrees = None if box is None else pointing_game(attribution, box[None])
        row = {"image": Path(image_set.paths[i]).name, "true": image_set.class_names[image_set.labels[i]],
               "p(fractured)": p, "detector box": box, "detector confidence": float(found[0, 4]) if len(found) else np.nan,
               "xai box": xai_box, "explanation agrees": agrees, "verdict": verdict(p, box is not None, threshold)}
        row["report"] = write_report(row, image_set.sizes[i], image_set.images.shape[1], method, threshold)
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(row, original_size, size, method, threshold=0.5):
    """A few lines a radiologist can read; boxes in pixels of the original X-ray."""
    p = row["p(fractured)"]
    lines = [f"{row['image']}: {row['verdict'].upper()}",
             f"classifier: {'fractured' if p >= threshold else 'not fractured'} (p = {p:.2f})"]
    box = row["detector box"]
    if box is None:
        lines.append("detector: no fracture box")
    else:
        sx, sy = original_size[0] / size, original_size[1] / size
        lines.append(f"detector: fracture at x {box[0] * sx:.0f}-{box[2] * sx:.0f}, y {box[1] * sy:.0f}-{box[3] * sy:.0f} px"
                     f" (confidence {row['detector confidence']:.2f})")
    if row["explanation agrees"] is not None:
        lines.append(f"explanation ({method}): the classifier looks "
                     + ("inside the detected box" if row["explanation agrees"] else "outside the detected box: check it"))
    if row["verdict"] == "needs review":
        lines.append("classifier and detector disagree: needs a radiologist's review")
    return "\n".join(lines)


def outcomes(cases):
    """How many X-rays of each true class end in each verdict."""
    return pd.crosstab(cases["true"], cases["verdict"]).reindex(columns=list(VERDICTS), fill_value=0)


def scores(cases):
    """The pipeline's numbers: how often it decides, how often it is right when it decides, what it misses."""
    fractured = cases["true"] == "fractured"
    decided = cases["verdict"] != "needs review"
    agreement = cases["explanation agrees"].dropna().astype(bool)
    return {
        "needs review (%)": 100 * float((~decided).mean()),
        "accuracy when decided (%)": 100 * float(((cases["verdict"] == "fracture") == fractured)[decided].mean())
        if decided.any() else float("nan"),
        "fractures missed (%)": 100 * float((cases["verdict"][fractured] == "no fracture").mean())
        if fractured.any() else float("nan"),
        "explanation inside the detector box (%)": 100 * float(agreement.mean()) if len(agreement) else float("nan"),
    }
