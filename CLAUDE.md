# CLAUDE.md

Exam project (Visual Intelligence, UniVR 2025/26): bone X-ray fracture classification, CNN vs ScatNet
(+ ResNet18), six XAI methods, one (Occlusion) from scratch. The exam PDF's requirements are listed in
`.claude/skills/exam-checklist/SKILL.md`.

## Layout and rules

- All code in flat modules under `src/` (see `src/__init__.py`); `notebooks/main.ipynb` only has a
  setup cell, a **Settings** cell and short calls to `src` - never define functions in it.
- Every model ends with `models.Classifier` (exam: same classifier, only `in_features` differs).
  Keep it that way; `tests/test_models.py` enforces it.
- Images are grey, 1 channel, normalised to [-1, 1]; black = `data.BLACK` = -1 is the XAI baseline.
- F1 / precision / recall are for class 0 = `fractured` (`training.POSITIVE`).
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

The real dataset is only on Kaggle (`bmadushanirodrigo/fracture-multi-region-x-ray-data`); a cloud
sandbox may not reach kaggle.com, so training happens on the Kaggle kernel (`kaggle-run` skill).
