# CLAUDE.md

Exam project (Visual Intelligence, UniVR 2025/26): bone X-ray fracture classification on FracAtlas, CNN vs
ScatNet (+ ResNet18), six XAI methods, one (Occlusion) from scratch, XAI maps turned into fracture boxes
scored against the radiologists' boxes (`localize.py`), and a full pipeline after Linda (2025): best classifier
+ YOLOv8 (`detect.py`) + best XAI method -> verdict and report (`pipeline.py`). The exam PDF's requirements are listed in
`.claude/skills/exam-checklist/SKILL.md`.

## Layout and rules

- All code in flat modules under `src/` (see `src/__init__.py`); `notebooks/main.ipynb` only has a
  setup cell, a **Settings** cell and short calls to `src` - never define functions in it.
- Every model ends with `models.Classifier` (exam: same classifier, only `in_features` differs).
  Keep it that way; `tests/test_models.py` enforces it.
- Images are grey, 1 channel, normalised to [-1, 1]; black = `data.BLACK` = -1 is the XAI baseline.
- F1 / precision / recall are for class 0 = `fractured` (`training.POSITIVE`). Classes are unbalanced
  (1 in 6 fractured): the loss is weighted (`training.class_weights`).
- Fracture boxes are `(x0, y0, x1, y1)` in pixels of the 224x224 image (`ImageSet.boxes`, from the YOLO files).
- `ultralytics` is imported only inside `detect.py` functions; unit tests use a fake `ultralytics`
  (`tests/test_detect.py`), the notebook test needs the real one (QUICK builds YOLO from `yolov8n.yaml`).
- Figures: one function per figure in `plots.py`, fixed colours (`MODEL_COLORS`, `METHOD_COLORS`),
  always `save_to=FIG_DIR / "<name>.png"`; the slides include them by that name.
- Numbers in the slides come only from `results/latex/numbers.tex` (`report.latex_macros`); add a
  fallback in `presentation/defaults.tex` for every new macro.
- Kymatio 0.3 needs `utils.patch_scipy_for_kymatio()` before `import kymatio` (SciPy >= 1.17).

## Commands

```bash
pytest -m "not slow"                  # fast tests (CPU, synthetic data)
pytest -m slow                        # whole notebook on synthetic data (~4 min)
./run.sh push [--quick] | status | get | slides
make -C presentation                  # slides from results/
python .claude/skills/exam-checklist/scripts/check_exam.py
```

The real dataset is only on Kaggle (`mahmudulhasantasin/fracatlas-original-dataset`); a cloud
sandbox may not reach kaggle.com, so training happens on the Kaggle kernel (`kaggle-run` skill).
