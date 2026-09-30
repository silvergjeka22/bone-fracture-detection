"""Full pipeline after Linda (2025): classifier + YOLO box + explanation -> verdict and report per X-ray.

Verdict: 'fracture' / 'no fracture' when classifier and detector agree, otherwise 'needs review'.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .localize import heatmap_to_box, pointing_game
from .xai import DEFAULT_PARAMS, METHODS

VERDICTS = ("fracture", "no fracture", "needs review")


def verdict(p_fractured, has_box, threshold=0.5):
    """'fracture' / 'no fracture' when classifier and detector agree, 'needs review' when they do not."""
    if p_fractured >= threshold and has_box:
        return "fracture"
    if p_fractured < threshold and not has_box:
        return "no fracture"
    return "needs review"


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
