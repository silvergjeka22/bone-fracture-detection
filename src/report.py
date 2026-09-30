"""Export the results: results/summary.json and the LaTeX numbers and tables used by the slides."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

MACRO_MODEL = {"cnn": "CNN", "scatnet": "Scat", "resnet18": "Res"}
MACRO_METHOD = {"Saliency": "Sal", "Integrated Gradients": "IG", "Guided Backprop": "GBP", "Grad-CAM": "GradCAM",
                "Occlusion": "Occ", "LIME": "LIME", "Occlusion (ours)": "OccOurs", "Random": "Random"}
MODEL_LABELS = {"cnn": "CNN", "scatnet": "ScatNet", "resnet18": "ResNet18"}


def build_summary(data_table, duplicate_table, cv, final, test, best, select_by, mcnemar, xai_results, settings,
                  localization=None, pipeline=None):
    """All results in one JSON-friendly dict."""
    summary = {
        "settings": settings,
        "dataset": json.loads(data_table.to_json(orient="index")),
        "duplicates": json.loads(duplicate_table.to_json(orient="index")),
        "best_model": best, "select_by": select_by,
        "models": {}, "mcnemar": json.loads(mcnemar.to_json(orient="index")), "xai": {},
    }
    for name in test:
        t = test[name]
        summary["models"][name] = {
            "cv_mean": cv[name]["mean"], "cv_std": cv[name]["std"],
            "test": {k: t[k] for k in ("accuracy", "f1", "precision", "recall", "specificity", "auc")},
            "test_accuracy_ci": t["accuracy_ci"], "confusion_matrix": t["confusion_matrix"],
            "parameters": final[name]["parameters"], "best_epoch": final[name]["best_epoch"],
            "seconds_per_epoch": final[name]["seconds_per_epoch"], "ms_per_image": t["ms_per_image"],
        }
    for name, r in xai_results.items():
        if name == "scratch_vs_captum":
            summary["scratch_vs_captum"] = json.loads(r.to_json(orient="index"))
            continue
        summary["xai"][name] = {"deletion_auc": r["deletion"]["deletion AUC (mean)"].to_dict(),
                                "seconds_per_image": r["seconds"]}
    if localization:
        summary["localization"] = {name: json.loads(table.to_json(orient="index"))
                                   for name, table in localization.items()}
    if pipeline:
        summary["pipeline"] = json.loads(json.dumps(pipeline, default=float))
    return summary


def save_summary(summary, results_dir):
    path = Path(results_dir) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return path


def _pct(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def latex_macros(summary):
    """\\newcommand lines for every number the slides quote."""
    lines = []

    def add(name, value):
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    best = summary["best_model"]
    settings = summary.get("settings", {})
    add("Epochs", settings.get("epochs", "--"))
    add("KFolds", settings.get("k_folds", "--"))
    add("BestModel", MODEL_LABELS.get(best, best))
    add("BestModelKey", best)
    add("SelectBy", {"f1": "F1", "accuracy": "accuracy"}.get(summary["select_by"], summary["select_by"]))
    for split, row in summary["dataset"].items():
        add(f"N{split.capitalize()}", f"{int(row['total']):,}".replace(",", "{,}"))
    if "fractured" in summary["dataset"].get("test", {}):
        add("NTestFractured", int(summary["dataset"]["test"]["fractured"]))
    dup = summary["duplicates"].get("dataset")
    if dup:
        add("DupImages", int(dup["with a near-duplicate"]))
        add("DupPct", dup["%"])
        add("DupGroups", int(dup["duplicate groups"]))
    for name, m in summary["models"].items():
        tag = MACRO_MODEL.get(name, re.sub(r"[^A-Za-z]", "", name))
        add(f"CVAcc{tag}", f"{_pct(m['cv_mean']['accuracy'])} $\\pm$ {_pct(m['cv_std']['accuracy'])}")
        add(f"CVFone{tag}", f"{_pct(m['cv_mean']['f1'])} $\\pm$ {_pct(m['cv_std']['f1'])}")
        add(f"TestAcc{tag}", _pct(m["test"]["accuracy"]))
        add(f"TestAccCI{tag}", f"[{_pct(m['test_accuracy_ci'][0])}, {_pct(m['test_accuracy_ci'][1])}]")
        add(f"TestFone{tag}", _pct(m["test"]["f1"]))
        add(f"TestRecall{tag}", _pct(m["test"]["recall"]))
        add(f"TestAUC{tag}", _pct(m["test"]["auc"]))
        add(f"Params{tag}", f"{m['parameters'] / 1e6:.1f}M")
        add(f"SecEpoch{tag}", f"{m['seconds_per_epoch']:.0f}")
    for name, x in summary["xai"].items():
        tag = MACRO_MODEL.get(name, re.sub(r"[^A-Za-z]", "", name))
        ranked = [k for k in x["deletion_auc"] if k not in ("Random", "Occlusion (ours)")]
        add(f"BestXAI{tag}", ranked[0] if ranked else "--")
        for method, auc in x["deletion_auc"].items():
            add(f"Del{tag}{MACRO_METHOD.get(method, re.sub(r'[^A-Za-z]', '', method))}", f"{auc:.3f}")
    for pair, row in summary["mcnemar"].items():
        a, b = pair.split(" vs ")
        add(f"McNemar{MACRO_MODEL.get(a, a)}vs{MACRO_MODEL.get(b, b)}", f"{row['p-value']:.3g}")
    if "scratch_vs_captum" in summary:
        worst = max(v["max |difference|"] for v in summary["scratch_vs_captum"].values())
        corr = min(v["min Pearson r"] for v in summary["scratch_vs_captum"].values())
        add("ScratchMaxDiff", f"{worst:.1e}")
        add("ScratchMinCorr", f"{corr:.4f}")
    localization = summary.get("localization", {})
    for name, rows in localization.items():
        tag = MACRO_MODEL.get(name, re.sub(r"[^A-Za-z]", "", name))
        methods = {k: v for k, v in rows.items() if k != "Random"}
        best = max(methods, key=lambda k: methods[k]["hit rate (%)"]) if methods else None
        add(f"BestBox{tag}", best or "--")
        add(f"BestHit{tag}", f"{methods[best]['hit rate (%)']:.0f}" if best else "--")
        for method, row in methods.items():
            add(f"Hit{tag}{MACRO_METHOD.get(method, re.sub(r'[^A-Za-z]', '', method))}", f"{row['hit rate (%)']:.0f}")
    random = [rows["Random"]["hit rate (%)"] for rows in localization.values() if "Random" in rows]
    if random:  # the same fractured test X-rays for every model
        add("HitRandom", f"{random[0]:.0f}")
    pipe = summary.get("pipeline")
    if pipe:
        det, boxes, cases = pipe["detector"], pipe["detector_boxes"], pipe["cases"]
        add("YoloMapFifty", _pct(det["mAP@0.5"]))
        add("YoloMap", _pct(det["mAP@0.5:0.95"]))
        add("YoloPrecision", _pct(det["precision"]))
        add("YoloRecall", _pct(det["recall"]))
        add("YoloHit", f"{boxes['hit rate (%)']:.0f}")
        add("YoloIoU", f"{boxes['IoU']:.2f}")
        add("BoxMethod", pipe["box_method"])
        for macro, key in (("ReviewPct", "needs review (%)"), ("DecidedAcc", "accuracy when decided (%)"),
                           ("MissedPct", "fractures missed (%)"), ("AgreePct", "explanation inside the detector box (%)")):
            value = cases[key]
            add(macro, "--" if value is None or np.isnan(value) else f"{value:.0f}")
    return "\n".join(lines) + "\n"


def to_latex(table, index_name=""):
    """DataFrame -> booktabs tabular (no pandas LaTeX dependency on jinja2)."""
    def esc(v):
        if isinstance(v, (float, np.floating)):
            text = "--" if np.isnan(v) else f"{v:.1f}" if abs(v) >= 1 else f"{v:.3f}"
        else:
            text = str(v)
        return text.replace("\\", "\\textbackslash{}").replace("%", "\\%").replace("_", "\\_").replace(
            "&", "\\&").replace("±", "$\\pm$").replace("#", "\\#")

    cols = list(table.columns)
    lines = [f"\\begin{{tabular}}{{l{'r' * len(cols)}}}", "\\toprule",
             " & ".join([esc(index_name)] + [esc(c) for c in cols]) + " \\\\", "\\midrule"]
    for idx, row in table.iterrows():
        lines.append(" & ".join([esc(MODEL_LABELS.get(idx, idx))] + [esc(v) for v in row.values]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def export_latex(summary, tables, results_dir):
    """Write numbers.tex and one .tex file per table in results/latex/."""
    out = Path(results_dir) / "latex"
    out.mkdir(parents=True, exist_ok=True)
    (out / "numbers.tex").write_text(latex_macros(summary))
    for name, table in tables.items():
        (out / f"{name}.tex").write_text(to_latex(pd.DataFrame(table)))
    return out


def key_findings(summary):
    """The main results in a few printed lines (a starting point for the discussion)."""
    models = summary["models"]
    best = summary["best_model"]
    lines = [f"Best model (highest CV {summary['select_by']}): {MODEL_LABELS.get(best, best)}"]
    for name, m in models.items():
        lines.append(f"  {MODEL_LABELS.get(name, name):9s} CV acc {_pct(m['cv_mean']['accuracy'])}% | "
                     f"test acc {_pct(m['test']['accuracy'])}% | "
                     f"test F1 {_pct(m['test']['f1'])}% | AUC {_pct(m['test']['auc'])}% | "
                     f"{'meets' if m['test']['accuracy'] >= 0.75 else 'BELOW'} the 75% target")
    for pair, row in summary["mcnemar"].items():
        lines.append(f"  McNemar {pair}: p = {row['p-value']}")
    for name, x in summary["xai"].items():
        ranked = [k for k in x["deletion_auc"] if k != "Occlusion (ours)"]
        lines.append(f"  most faithful XAI on {MODEL_LABELS.get(name, name)} (deletion test): "
                     + " < ".join(f"{k} {x['deletion_auc'][k]:.3f}" for k in ranked))
    if "scratch_vs_captum" in summary:
        worst = max(v["max |difference|"] for v in summary["scratch_vs_captum"].values())
        lines.append(f"  Occlusion from scratch vs Captum: max |difference| = {worst:.1e}")
    for name, rows in summary.get("localization", {}).items():
        lines.append(f"  fracture found (pointing game) on {MODEL_LABELS.get(name, name)}: "
                     + ", ".join(f"{k} {v['hit rate (%)']:.0f}%" for k, v in rows.items()))
    pipe = summary.get("pipeline")
    if pipe:
        det, cases = pipe["detector"], pipe["cases"]
        lines.append(f"  YOLO on the fractured test X-rays: mAP@0.5 {_pct(det['mAP@0.5'])}% (FracAtlas paper: 56.2%), "
                     f"hit rate {pipe['detector_boxes']['hit rate (%)']:.0f}%")
        lines.append("  full pipeline (" + pipe["box_method"] + "): "
                     + ", ".join(f"{k} {v:.0f}" for k, v in cases.items()))
    text = "\n".join(lines)
    print(text)
    return text


def export(results_dir, settings, data_table, duplicate_table, cv, final, test, best, select_by, mcnemar,
           deletion, seconds, scratch_table, comparison, localization=None, pipeline=None):
    """Build and save summary.json and the LaTeX macros/tables of the slides; returns the summary."""
    from .training import cv_table, test_table

    xai_results = {n: {"deletion": deletion[n], "seconds": seconds[n]} for n in deletion}
    xai_results["scratch_vs_captum"] = scratch_table
    summary = build_summary(data_table, duplicate_table, cv, final, test, best, select_by, mcnemar,
                            xai_results, settings, localization, pipeline)
    save_summary(summary, results_dir)
    tables = {
        "dataset": data_table, "duplicates": duplicate_table, "cv": cv_table(cv), "test": test_table(test),
        "mcnemar": mcnemar,
        "deletion": pd.DataFrame({MODEL_LABELS.get(n, n): d["deletion AUC (mean)"] for n, d in deletion.items()}),
        "comparison": comparison[["parameters (M)", "s / epoch", "ms / image"]].round(1),
    }
    if localization:  # hit rates in %, as whole numbers
        hits = pd.DataFrame({MODEL_LABELS.get(n, n): t["hit rate (%)"] for n, t in localization.items()})
        tables["localization"] = hits.apply(lambda col: col.map(lambda v: "--" if pd.isna(v) else f"{v:.0f}"))
    export_latex(summary, tables, results_dir)
    print(f"saved {Path(results_dir) / 'summary.json'} and {Path(results_dir) / 'latex'}")
    return summary
