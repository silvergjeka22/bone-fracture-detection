"""Check the results of a run against the exam requirements.

    python .claude/skills/exam-checklist/scripts/check_exam.py [results_dir]

Prints one line per requirement (OK / MISSING / FAIL) and exits with 1 if any requirement fails.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

REQUIRED_FIGURES = {
    "learning curves (train + val together)": "learning_curves",
    "CNN filters": "filters_cnn",
    "ScatNet filters": "filters_scatnet",
    "XAI overlays on the CNN": "xai_cnn",
    "XAI overlays on ScatNet": "xai_scatnet",
    "scratch vs Captum figure": "occlusion_scratch_vs_captum",
    "confusion matrices": "confusion_matrices",
}
SIX_METHODS = {"Saliency", "Integrated Gradients", "Guided Backprop", "Grad-CAM", "Occlusion", "LIME"}


def main():
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
    rows = []

    def check(name, ok, detail=""):
        rows.append((name, ok, detail))

    from src import models, xai

    layouts = [models.classifier_layout(models.build_model(n, 64, pretrained=False)) for n in ("cnn", "scatnet")]
    check("4. CNN and ScatNet share the same classifier", layouts[0] == layouts[1])
    check("8. six XAI methods implemented", set(xai.METHODS) == SIX_METHODS, ", ".join(xai.METHODS))
    check("8. one method implemented from scratch", (ROOT / "src" / "occlusion_scratch.py").exists())

    summary_path = results / "summary.json"
    if not summary_path.exists():
        check("results of a run", False, f"{summary_path} missing: run the notebook (./run.sh push, ./run.sh get)")
    else:
        s = json.loads(summary_path.read_text())
        quick = s.get("settings", {}).get("quick")
        check("full run (not QUICK)", not quick)
        check("2. binary dataset", all(len([k for k in row if k not in ("total", "median width (px)",
              "median height (px)", "mean intensity") and not k.startswith("%")]) == 2 for row in s["dataset"].values()))
        check("3. separate test split", "test" in s["dataset"] and "train" in s["dataset"])
        for name in ("cnn", "scatnet"):
            m = s["models"].get(name)
            if m is None:
                check(f"5/7. {name} trained", False)
                continue
            check(f"5. {name}: mean CV accuracy and mean CV F1",
                  "accuracy" in m["cv_mean"] and "f1" in m["cv_mean"],
                  f"acc {m['cv_mean']['accuracy']:.3f}, F1 {m['cv_mean']['f1']:.3f}")
            check(f"7. {name}: test accuracy >= 75%", m["test"]["accuracy"] >= 0.75, f"{m['test']['accuracy']:.3f}")
            methods = set(s["xai"].get(name, {}).get("deletion_auc", {})) - {"Random"}
            expected = SIX_METHODS - ({"Grad-CAM"} if name == "scatnet" else set())
            check(f"8. XAI methods computed on {name}", expected <= methods, ", ".join(sorted(methods)))
        check("9. scratch vs Captum compared", "scratch_vs_captum" in s,
              f"max |diff| {max(v['max |difference|'] for v in s['scratch_vs_captum'].values()):.1e}"
              if "scratch_vs_captum" in s else "")
        check("11. at least 2 images per class explained", s.get("settings", {}).get("xai_per_class", 0) >= 2)
        check("best model chosen by cross-validation", bool(s.get("best_model")), s.get("best_model", ""))
    for name, stem in REQUIRED_FIGURES.items():
        check(f"figure: {name}", (results / "figures" / f"{stem}.png").exists())
    check("LaTeX numbers for the slides", (results / "latex" / "numbers.tex").exists())
    check("presentation built", (ROOT / "presentation" / "main.pdf").exists())

    for name, ok, detail in rows:
        print(f"{'OK  ' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    failed = sum(not ok for _, ok, _ in rows)
    print(f"\n{len(rows) - failed}/{len(rows)} requirements met")
    print("Manual: discussion vs numbers, report (6-8 pages), presentation timing (<= 12 min).")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
