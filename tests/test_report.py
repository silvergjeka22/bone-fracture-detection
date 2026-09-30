import json

import pandas as pd

from src import report


def fake_summary():
    metrics = {"accuracy": 0.9, "f1": 0.88, "precision": 0.9, "recall": 0.86, "specificity": 0.93, "auc": 0.95}
    model = {"cv_mean": metrics, "cv_std": {k: 0.01 for k in metrics}, "test": metrics,
             "test_accuracy_ci": [0.87, 0.93], "confusion_matrix": [[1, 0], [0, 1]], "parameters": 26_000_000, "best_epoch": 12, "seconds_per_epoch": 30.0, "ms_per_image": 2.0}
    return {
        "settings": {}, "best_model": "cnn", "select_by": "f1",
        "dataset": {"train": {"total": 2913}, "val": {"total": 585}, "test": {"total": 585, "fractured": 103}},
        "duplicates": {"dataset": {"with a near-duplicate": 100, "%": 2.4, "duplicate groups": 40}},
        "models": {"cnn": model, "scatnet": model},
        "mcnemar": {"cnn vs scatnet": {"p-value": 0.03}},
        "xai": {"cnn": {"deletion_auc": {"Occlusion": 0.2, "Saliency": 0.3, "Random": 0.5}, "seconds_per_image": {}}},
        "scratch_vs_captum": {"cnn": {"max |difference|": 1e-7, "min Pearson r": 1.0}},
        "localization": {name: {"Occlusion": {"hit rate (%)": 62.4, "IoU": 0.21, "box size (%)": 4.0},
                                "Saliency": {"hit rate (%)": 40.0, "IoU": 0.1, "box size (%)": 2.0},
                                "Random": {"hit rate (%)": 3.2, "IoU": None, "box size (%)": None}}
                         for name in ("cnn", "scatnet")},
    }


def test_latex_macros():
    tex = report.latex_macros(fake_summary())
    for macro in ("\\BestModel}{CNN}", "\\TestAccCNN}{90.0}", "\\TestAccScat}", "\\CVAccCNN}{90.0 $\\pm$ 1.0}",
                  "\\NTrain}{2{,}913}", "\\NTestFractured}{103}", "\\DupImages}{100}", "\\BestXAICNN}{Occlusion}",
                  "\\ScratchMaxDiff}", "\\BestBoxCNN}{Occlusion}", "\\BestHitCNN}{62}", "\\HitCNNSal}{40}",
                  "\\HitRandom}{3}"):
        assert macro in tex
    lines = tex.strip().splitlines()
    assert all(line.startswith("\\newcommand{\\") for line in lines)
    names = [line.split("}")[0] for line in lines]
    assert len(names) == len(set(names)), "a macro defined twice breaks the slides"


def test_to_latex_escapes():
    table = pd.DataFrame({"acc %": ["90.0 ± 1.0"], "F1_score": [0.5], "recall": [56.6]}, index=["cnn"])
    tex = report.to_latex(table)
    assert "\\%" in tex and "$\\pm$" in tex and "F1\\_score" in tex and "CNN" in tex
    assert "& 0.500 & 56.6 \\\\" in tex  # percentages with 1 decimal, small values with 3
    assert tex.startswith("\\begin{tabular}") and "\\bottomrule" in tex


def test_export_and_key_findings(tmp_path):
    summary = fake_summary()
    report.save_summary(summary, tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text())["best_model"] == "cnn"
    out = report.export_latex(summary, {"cv": pd.DataFrame({"a": [1.0]}, index=["cnn"])}, tmp_path)
    assert (out / "numbers.tex").exists() and (out / "cv.tex").exists()
    text = report.key_findings(summary)
    assert "meets the 75% target" in text and "Occlusion 0.200" in text and "Occlusion 62%" in text
