"""Save every result in results/summary.json and print the key findings."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {"cnn": "CNN", "scatnet": "ScatNet", "resnet18": "ResNet18"}
METRICS = ("accuracy", "f1", "precision", "recall", "specificity", "auc")


def _json(value):
    """DataFrames, numpy numbers and nested dicts -> plain JSON values."""
    if isinstance(value, pd.DataFrame):
        return json.loads(value.to_json(orient="index"))
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value.item() if hasattr(value, "item") else value


def build_summary(settings, dataset, duplicates, cv, final, test, best, select_by, mcnemar, deletion, seconds,
                  scratch, localization=None, guided=None, pipeline=None):
    """All results in one JSON-friendly dict (guided models have no cross-validation)."""
    models = {name: {"cv_mean": cv.get(name, {}).get("mean"), "cv_std": cv.get(name, {}).get("std"),
                     "test": {k: t[k] for k in METRICS}, "test_accuracy_ci": t["accuracy_ci"],
                     "confusion_matrix": t["confusion_matrix"], "parameters": final[name]["parameters"],
                     "best_epoch": final[name]["best_epoch"]} for name, t in test.items()}
    xai = {n: {"deletion_auc": d["deletion AUC (mean)"].to_dict(), "seconds_per_image": seconds.get(n, {})}
           for n, d in deletion.items()}
    return _json({"settings": settings, "dataset": dataset, "duplicates": duplicates, "best_model": best,
                  "select_by": select_by, "models": models, "mcnemar": mcnemar, "xai": xai,
                  "scratch_vs_captum": scratch, "localization": localization or {}, "guided": guided,
                  "pipeline": pipeline})


def export(results_dir, **parts):
    """Build and save results/summary.json; returns the summary."""
    summary = build_summary(**parts)
    path = Path(results_dir) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1, default=str))
    print(f"saved {path}")
    return summary


def _pct(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def _label(name):
    """'resnet18_box_contrast' -> 'ResNet18 + box + contrast'."""
    base, *extra = name.split("_")
    return " + ".join([LABELS[base], *extra]) if base in LABELS else name


def key_findings(summary):
    """The main results in a few printed lines."""
    label = _label
    lines = [f"Best model (highest CV {summary['select_by']}): {label(summary['best_model'])}"]
    for name, m in summary["models"].items():
        cv = f"CV F1 {_pct(m['cv_mean']['f1'])}% | " if m["cv_mean"] else ""
        t = m["test"]
        lines.append(f"  {label(name):26s} {cv}test acc {_pct(t['accuracy'])}% F1 {_pct(t['f1'])}% "
                     f"fractures found {_pct(t['recall'])}% AUC {_pct(t['auc'])}%"
                     + ("" if t["accuracy"] >= 0.75 else "  (BELOW the 75% target)"))
    for pair, row in summary["mcnemar"].items():
        lines.append(f"  McNemar {pair}: p = {row['p-value']}")
    for name, x in summary["xai"].items():
        ranked = [k for k in x["deletion_auc"] if k != "Occlusion (ours)"]
        lines.append(f"  deletion test, {label(name)} (lower = more faithful): "
                     + " < ".join(f"{k} {x['deletion_auc'][k]:.3f}" for k in ranked))
    if summary.get("scratch_vs_captum"):
        worst = max(v["max |difference|"] for v in summary["scratch_vs_captum"].values())
        lines.append(f"  Occlusion from scratch vs Captum: max |difference| = {worst:.1e}")
    for name, rows in summary["localization"].items():
        lines.append(f"  hit rate, {label(name)}: " + ", ".join(f"{k} {v['hit rate (%)']:.0f}%" for k, v in rows.items()))
    for name, row in (summary.get("guided") or {}).items():
        lines.append(f"  guided comparison, {name}: F1 {row['test F1 (%)']:.1f}%, mean hit rate "
                     f"{row['hit rate, mean of methods (%)']:.0f}%, best {row['best method']} "
                     f"{row['hit rate, best method (%)']:.0f}%")
    pipe = summary.get("pipeline")
    if pipe:
        lines.append(f"  YOLO: mAP@0.5 {_pct(pipe['detector']['mAP@0.5'])}% (FracAtlas paper 56.2%), "
                     f"hit rate {pipe['detector_boxes']['hit rate (%)']:.0f}%")
        lines.append(f"  pipeline ({label(pipe['classifier'])} + {pipe['box_method']} + YOLO): "
                     + ", ".join(f"{k} {v:.0f}" for k, v in pipe["cases"].items()))
    text = "\n".join(lines)
    print(text)
    return text
