---
name: kaggle-run
description: Run this project's notebook (notebooks/main.ipynb) on a Kaggle GPU kernel and bring the results back - privately upload an allowlisted local source snapshot with run.sh, check status, download output into results/, rebuild slides, and debug failed Kaggle runs (dataset not found, wrong source snapshot, kymatio/scipy error, missing ResNet18 weights, 12-hour limit). Use it whenever the user wants to train, re-run, get results, "run it on Kaggle", check a Kaggle run, or asks why the Kaggle kernel failed, even if they don't say "skill".
---

# Run the experiment on Kaggle

The notebook is designed to run top to bottom on a Kaggle GPU kernel, launched from a terminal
(the pattern of the `macura-drone` project): `kernel-metadata.json` describes the kernel, `run.sh`
wraps the Kaggle CLI, `src/bootstrap.py` prepares the kernel.

## How the pieces fit

| file | role |
|---|---|
| `kernel-metadata.json` | kernel id (`<kaggle-user>/bone-fracture-detection`), GPU (T4) + internet on, **dataset attached**: `bmadushanirodrigo/fracture-multi-region-x-ray-data` |
| `run.sh` | Git Bash / POSIX entry point for `scripts/kaggle_run.py` |
| `scripts/kaggle_run.py` | allowlists local source, creates/reuses an immutable private code Dataset, attaches it and starts the job |
| notebook cell 1 | verifies and reconstructs the attached code snapshot under `/tmp`; no GitHub credentials |
| `src/bootstrap.setup()` | pip-installs kymatio + captum only, finds the dataset under `/kaggle/input`, results -> `/kaggle/working/results` (the kernel output), image cache -> `/tmp` |

`run.sh push` never edits the repo notebook. It stages a copy with the code checksum and `QUICK`
filled in, attaches only the required local source files, and refuses credential-like content.

## Workflow

1. One-time: activate `bone-fracture`, install `requirements.txt`, move a legacy
   `~/.kaggle/kaggle.json` to `.backup` (legacy credentials can override OAuth), then run
   `kaggle auth login --force`, sign in to the account named in `kernel-metadata.json`, and accept
   the requested Dataset/Notebook permissions. Use `--no-launch-browser` when necessary.
2. Optional offline preview: `./run.sh push --quick --dry-run` (no authentication or upload).
3. Health check (~5 min): `./run.sh push --quick` -> `./run.sh status` until `complete` -> `./run.sh get`.
4. Full run (~2-3 h on a T4, 12 h is the limit): `./run.sh push`. Once accepted, the PC can be switched off.
5. `./run.sh get` downloads to a new timestamped folder in `./out` and copies completed `results` into `./results`
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
| code Dataset rejected / not private | check the Kaggle username in `kernel-metadata.json`; the launcher never submits a kernel until it confirms privacy |
| upload reaches 100%, then `403 Forbidden` | Kaggle accepted the temporary blob but denied Dataset creation; move legacy `kaggle.json` aside, run `kaggle auth login --force`, and accept Dataset/Notebook permissions |
| wrong local branch | switch branches locally first; the launcher uploads the checked-out local files and records their checksum |
| `cannot import name 'sph_harm'` | SciPy >= 1.17 with Kymatio 0.3: `src/utils.patch_scipy_for_kymatio()` must run before `import kymatio` (models.py and plots.py do it) |
| ResNet18 weights download fails | internet disabled: `enable_internet: true` |
| `PyTorch cannot run on Tesla P100` / `no kernel image is available` | recent PyTorch dropped the P100: keep `machine_shape: NvidiaTeslaT4` (or pick T4 in the kernel settings) |
| CUDA out of memory | lower `BATCH_SIZE` in the Settings cell (ScatNet's classifier has 42M weights) |
| killed after 12 h, no output | a commit that exceeds the limit saves nothing: lower `EPOCHS`/`K_FOLDS` or drop `resnet18` from `MODELS` |
| `kaggle: command not found` | `pip install kaggle` on the machine that runs `run.sh` |

The session's own sandbox may block kaggle.com: then the run must be launched by the user from
their machine; say so instead of retrying.
