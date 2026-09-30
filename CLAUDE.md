# CLAUDE.md

Exam project (Visual Intelligence, UniVR 2025/26): fracture classification on FracAtlas, CNN vs ScatNet
(+ ResNet18), six XAI methods (Occlusion also from scratch), XAI maps -> fracture boxes (`localize.py`), and a
pipeline after Linda (2025): best classifier + YOLOv8 (`detect.py`) + best XAI -> report (`pipeline.py`).
Exam requirements: `.claude/skills/exam-checklist/SKILL.md`.

## Rules

- Code in flat `src/` modules; the notebook has only setup, a **Settings** cell and short calls (no functions).
- Keep it simple: one-line docstrings, few comments, no unused code or files.
- Every model ends with `models.Classifier` (same classifier; `tests/test_models.py` enforces it).
- Images: grey, 1 channel, [-1, 1]; black = `data.BLACK` = -1 is the XAI baseline.
- Class 0 = `fractured` is the positive class; the loss is weighted (1 in 6 fractured).
- Boxes are `(x0, y0, x1, y1)` in pixels of the 224x224 image (`ImageSet.boxes`).
- `ultralytics` is imported only inside `detect.py` functions (unit tests use a fake one).
- Figures: one function each in `plots.py`, saved as `FIG_DIR / "<name>.png"` (the slides use that name).
- Slide numbers only from `report.latex_macros`, each with a fallback in `presentation/defaults.tex`.
- `utils.patch_scipy_for_kymatio()` before `import kymatio`.

## Commands

```bash
pytest -m "not slow"                           # fast tests
pytest -m slow                                 # whole notebook on synthetic data
./run.sh push [--quick] | status | get | slides   # Kaggle (Windows: run.bat)
python .claude/skills/exam-checklist/scripts/check_exam.py
```

The dataset is on Kaggle (`mahmudulhasantasin/fracatlas-original-dataset`); training runs there (`kaggle-run` skill).
