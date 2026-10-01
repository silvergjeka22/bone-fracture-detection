---
name: kaggle-run
description: Run this project's notebook (notebooks/main.ipynb) on a Kaggle GPU and bring the results back with ./run.sh (macOS, Linux, Git Bash) or run.bat (Windows) - push, quick check, status, download into results/, and debug failed Kaggle runs (login, 403, dataset not found, kymatio/scipy error, missing weights, 12-hour limit). Use it whenever the user wants to train, re-run, get results, "run it on Kaggle", check a Kaggle run, or asks why the Kaggle kernel failed, even if they don't say "skill".
---

# Run the experiment on Kaggle

`run.sh` / `run.bat` call `scripts/kaggle_run.py`, which uses the `kaggle` command-line tool
(works with kaggle 1.7 on Python 3.9 and kaggle 2.x on 3.11+).

- `push [--quick] [--user NAME]`: packs `src/*.py` + `requirements.txt` into the first cell of the notebook
  (base64 zip, unpacked to `/tmp/bfd`), writes `.kaggle-build/` and runs `kaggle kernels push`. The kernel is
  `<logged-in username>/bone-fracture-detection`; settings (T4, internet, FracAtlas) from `kernel-metadata.json`.
  `--dry-run` only prepares `.kaggle-build/`.
- `status`, `get`: `kaggle kernels status / output` of the last pushed kernel (`.kaggle-kernel`); `get` saves
  into `out/run-<time>/` and copies `results/` into `./results` only if the run is complete.

## Workflow

1. `./run.sh push --quick`, then `./run.sh status` until complete, then `./run.sh get`.
2. `./run.sh push` (full run, ~2-3 h), later `./run.sh get`.
3. Commit `results/` (models, attributions, detector are gitignored), run the exam checklist.

The sandbox may not reach kaggle.com: then the user runs these commands; say so instead of retrying.

## Troubleshooting

| symptom | fix |
|---|---|
| login failed / 401 | new token: kaggle.com > Settings > API > Create New Token, `kaggle.json` in `~/.kaggle/` |
| 403 | account not phone-verified (GPU / internet), or `--user` differs from the logged-in account |
| wrong account | the account is the `kaggle.json` in `~/.kaggle/`; `kaggle config view` shows it |
| `No folder with images/Fractured` | FracAtlas not attached: check `dataset_sources` |
| `No YOLO annotation folder` | the attached FracAtlas copy lacks `Annotations/YOLO` |
| `cannot import name 'sph_harm'` | `utils.patch_scipy_for_kymatio()` must run before `import kymatio` |
| ultralytics / weights download fails | `enable_internet: true` in `kernel-metadata.json` |
| `no kernel image is available` | keep `machine_shape: NvidiaTeslaT4` (P100 is no longer supported) |
| CUDA out of memory | lower `BATCH_SIZE` in the Settings cell |
| killed after 12 h | lower `EPOCHS` / `K_FOLDS` or drop `resnet18` from `MODELS` |
| a run failed | `./run.sh get` and read the `.log` in `out/run-<time>/` |
