---
name: kaggle-run
description: Run this project's notebook (notebooks/main.ipynb) on a Kaggle GPU kernel and bring the results back - push with run.sh, check status, download the output into results/, rebuild the slides, and debug a failed Kaggle run (dataset not found, GITHUB_TOKEN secret, wrong branch, kymatio/scipy import error, missing ResNet18 weights, 12-hour limit). Use it whenever the user wants to train, re-run, get results, "run it on Kaggle", check a Kaggle run, or asks why the Kaggle kernel failed, even if they don't say "skill".
---

# Run the experiment on Kaggle

The notebook is designed to run top to bottom on a Kaggle GPU kernel, launched from a terminal
(the pattern of the `macura-drone` project): `kernel-metadata.json` describes the kernel, `run.sh`
wraps the Kaggle CLI, `src/bootstrap.py` prepares the kernel.

## How the pieces fit

| file | role |
|---|---|
| `kernel-metadata.json` | kernel id (`<kaggle-user>/bone-fracture-detection`), GPU + internet on, **dataset attached**: `bmadushanirodrigo/fracture-multi-region-x-ray-data` |
| `run.sh` | `push [--quick]`, `status`, `get`, `slides`, `stop` |
| notebook cell 1 | on Kaggle clones the private repo (`BRANCH`, default `main`) with the `GITHUB_TOKEN` secret into `/tmp` |
| `src/bootstrap.setup()` | pip-installs kymatio + captum only, finds the dataset under `/kaggle/input`, results -> `/kaggle/working/results` (the kernel output), image cache -> `/tmp` |

`run.sh push` never edits the repo notebook: it stages a copy with `BRANCH` and `QUICK` filled in.

## Workflow

1. One-time: `pip install kaggle`, `~/.kaggle/kaggle.json` (chmod 600), Kaggle Secret `GITHUB_TOKEN`
   (a GitHub token that can read the repo), own username in `kernel-metadata.json` `id`.
2. Code must be **pushed to GitHub** first: the kernel clones it. For an unmerged branch:
   `BRANCH=my-branch ./run.sh push`.
3. Health check (~5 min): `./run.sh push --quick` -> `./run.sh status` until `complete` -> `./run.sh get`.
4. Full run (~3-4 h on a P100/T4): `./run.sh push`. The PC can be switched off.
5. `./run.sh get` downloads into `./out` and copies `out/results` into `./results`
   (`summary.json`, `cv/`, `final/`, `figures/`, `latex/`, `models/*.pth`, `attributions/*.npz`).
6. `./run.sh slides` rebuilds `presentation/main.pdf` with the new numbers.
7. Commit `results/` **without** `models/` and `attributions/` (gitignored, too large), plus the new PDF.
8. Run the exam checklist skill (`python .claude/skills/exam-checklist/scripts/check_exam.py`).

To re-draw figures without retraining: open the notebook locally (or on Kaggle with the previous
output attached) with `TRAIN = False`; it reloads `results/models/*.pth` and the JSON logs.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `No folder with train/ and test/ found under /kaggle/input` | dataset not attached: check `dataset_sources`, or add it in the kernel's "Add data" panel |
| `UserSecretsClient ... GITHUB_TOKEN` error / git 403 | secret missing or not enabled for this kernel (Add-ons -> Secrets), or token without repo read access |
| old code runs | kernel cloned `main` but the change is on a branch: `BRANCH=<branch> ./run.sh push` |
| `cannot import name 'sph_harm'` | SciPy >= 1.17 with Kymatio 0.3: `src/utils.patch_scipy_for_kymatio()` must run before `import kymatio` (models.py and plots.py do it) |
| ResNet18 weights download fails | internet disabled: `enable_internet: true` |
| CUDA out of memory | lower `BATCH_SIZE` in the Settings cell (ScatNet's classifier has 42M weights) |
| killed after 12 h, no output | a commit that exceeds the limit saves nothing: lower `EPOCHS`/`K_FOLDS` or drop `resnet18` from `MODELS` |
| `kaggle: command not found` | `pip install kaggle` on the machine that runs `run.sh` |

The session's own sandbox may block kaggle.com: then the run must be launched by the user from
their machine; say so instead of retrying.
