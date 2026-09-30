---
name: exam-checklist
description: Check the bone-fracture project against every requirement of the Visual Intelligence exam (University of Verona 2025/2026) - CNN + ScatNet with the same classifier, k-fold CV with mean accuracy and mean F1, filters compared, test accuracy >= 75%, six XAI methods on both models, one implemented from scratch and compared with Captum, 2 images per class overlaid, discussion, report 6-8 pages, presentation <= 12 minutes. Use it before a submission, after a new Kaggle run, when the user asks "is everything done / what is missing / are we ready for the exam", or before changing anything that touches a requirement.
---

# Exam checklist

Run the automatic part first:

```bash
python .claude/skills/exam-checklist/scripts/check_exam.py            # uses results/
python .claude/skills/exam-checklist/scripts/check_exam.py out/results
```

It checks `results/summary.json`, the figures and the code, prints one line per requirement
and exits 1 if one fails. Then walk through the manual items below.

## Requirements (from the exam PDF) and where they live

| # | requirement | where | auto |
|---|---|---|---|
| 2 | binary dataset chosen by us | FracAtlas (with radiologists' fracture boxes); `data.py` | yes |
| 3 | train / test split | train / val / test folders; test used once (`training.evaluate_test`) | yes |
| 4 | CNN and ScatNet, **same final classifier** except input neurons | `models.Classifier`, `models.classifier_layout`, test `test_same_classifier_for_every_model` | yes |
| 5 | k-fold CV on the training set: **mean accuracy and mean F1** | `training.cross_validate` -> `results/cv/*.json` | yes |
| 6 | extract and compare the filters | `figures/filters_cnn.png`, `filters_scatnet.png`, `frequency_coverage.png` | yes |
| 7 | test on the test set, >= 75% accuracy | `summary.json` models.*.test.accuracy | yes |
| 8 | six XAI methods on CNN **and** ScatNet | `xai.METHODS` (Saliency, IG, Guided Backprop, Grad-CAM, Occlusion, LIME) | yes |
| 8 | one implemented from scratch | `src/occlusion_scratch.py` | yes |
| 9 | scratch vs Captum compared | `summary.json` scratch_vs_captum, `figures/occlusion_scratch_vs_captum.png` | yes |
| 10 | qualitative comparison per model | `figures/xai_cnn.png`, `xai_scatnet.png`, agreement, deletion test | yes |
| 11 | attributions **overlaid**, >= 2 images per class, both models | `plots.plot_attribution_grid(indices=SHOW)`, SHOW = 2 + 2 | yes |
| 12 | discussion: best method, method not usable on a model | notebook section 11; Grad-CAM not applicable to ScatNet | manual |
| - | learning curves train + val **in the same figure** | `figures/learning_curves.png` | yes |
| - | PyTorch + Captum + Kymatio | requirements.txt | yes |
| - | report 6-8 pages (8-10 for a pair), English, with figures | not in the repo yet | manual |
| - | presentation <= 12 min (15 for a pair), organised sections | `presentation/main.pdf` + `SPEAKER_NOTES.md` | manual |
| - | code must run at the oral exam | `pytest`, `./run.sh push --quick` | manual |

## Manual checks

- Read the discussion (notebook section 11, slide "Discussion and conclusions") against the
  actual numbers in `summary.json`; update any sentence the numbers contradict.
- If filters look noisy, the exam suggests more data augmentation (`data.AUGMENT`).
- Presentation: time it (target 11 minutes, `presentation/SPEAKER_NOTES.md`).
- Report: 6-8 pages; reuse the figures in `results/figures/` and tables in `results/latex/`.
- Nothing typed by hand in the slides: numbers come from `results/latex/numbers.tex`.
