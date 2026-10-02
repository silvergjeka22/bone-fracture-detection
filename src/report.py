"""Save every result in results/summary.json and print the key findings."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {"cnn": "CNN", "scatnet": "ScatNet", "resnet18": "ResNet18",
          "separate": "A: separate", "joint": "B: joint", "joint_xai": "C: joint + XAI"}
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
                  scratch, localization=None, joint=None, fusion=None, pipeline=None):
    """All results in one JSON-friendly dict (the joint versions have no cross-validation)."""
    models = {name: {"cv_mean": cv.get(name, {}).get("mean"), "cv_std": cv.get(name, {}).get("std"),
                     "test": {k: t[k] for k in METRICS}, "test_accuracy_ci": t["accuracy_ci"],
                     "threshold": t.get("threshold", 0.5),
                     "confusion_matrix": t["confusion_matrix"], "parameters": final[name]["parameters"],
                     "best_epoch": final[name]["best_epoch"]} for name, t in test.items()}
    xai = {n: {"deletion_auc": d["deletion AUC (mean)"].to_dict(), "seconds_per_image": seconds.get(n, {})}
           for n, d in deletion.items()}
    return _json({"settings": settings, "dataset": dataset, "duplicates": duplicates, "best_model": best,
                  "select_by": select_by, "models": models, "mcnemar": mcnemar, "xai": xai,
                  "scratch_vs_captum": scratch, "localization": localization or {}, "joint": joint,
                  "fusion": fusion, "pipeline": pipeline})


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


def key_findings(summary):
    """The main results in a few printed lines."""
    label = lambda name: LABELS.get(name, name) 
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
    joint = summary.get("joint") or {}
    for name, row in joint.get("comparison", {}).items():
        lines.append(f"  {name}: F1 {row['test F1 (%)']:.1f}%, fractures found {row['fractures found (%)']:.0f}%, "
                     f"detector AP@0.5 {row['detector AP@0.5 (%)']:.1f}% (hit {row['detector hit rate (%)']:.0f}%), "
                     f"Grad-CAM hit {row['Grad-CAM hit rate (%)']:.0f}%, best XAI {row['best XAI method']} "
                     f"{row['best XAI hit rate (%)']:.0f}%, blurring the fracture moves p(fractured) by "
                     f"{-row['p(fractured) drop, fracture blurred']:+.0f} points")
    for method, row in joint.get("hit_tests", {}).items():
        lines.append(f"  hit rate A -> C, {method}: {row['hits A']} -> {row['hits C']} X-rays (McNemar p = {row['p-value']})")
    fusion = summary.get("fusion") or {}
    if fusion:
        lines.append(f"  YOLO proposes, C decides (Grad-CAM weight {fusion['alpha']}, chosen on val):")
    for name, row in fusion.get("table", {}).items():
        lines.append(f"    {name}: box on the fracture {row['hit rate (%)']:.0f}% (McNemar p = "
                     f"{row['McNemar p vs the first row']}), AP@0.5 {row['AP@0.5, fractured X-rays (%)']:.1f}% "
                     f"(all X-rays {row['AP@0.5, all X-rays (%)']:.1f}%)")
    pipe = summary.get("pipeline")
    if pipe:
        lines.append(f"  YOLO: mAP@0.5 {_pct(pipe['detector']['mAP@0.5'])}% (FracAtlas paper 56.2%), "
                     f"hit rate {pipe['detector_boxes']['hit rate (%)']:.0f}%")
        lines.append(f"  pipeline ({label(pipe['classifier'])} + {pipe['box_method']} + YOLO): "
                     + ", ".join(f"{k} {v:.0f}" for k, v in pipe["cases"].items()))
    text = "\n".join(lines)
    print(text)
    return text
