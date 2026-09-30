# Bone Fracture Detection: CNN vs ScatNet, explained with XAI and fracture boxes

MSc in Artificial Intelligence, Visual Intelligence 2025/2026, University of Verona

Binary classification of X-rays (**fractured** / **not fractured**) on **FracAtlas**, where radiologists
drew a box around every fracture. A CNN trained from scratch and a wavelet Scattering Network (ScatNet,
Kymatio) are compared, with an ImageNet-pretrained ResNet18 as a reference; all three end with **the same
classifier**. The best model is chosen by cross-validation, tested once, its filters are compared with
ScatNet's wavelets, and the models are explained with **six XAI methods** (Captum), one of which
(Occlusion) is also implemented from scratch. Finally every explanation is turned into a **fracture box**
and checked against the radiologists' boxes: do the models look at the fracture?

Everything runs from one notebook, `notebooks/main.ipynb`, on a **Kaggle GPU** (`./run.sh push`).

## Exam requirements -> where

| requirement (exam PDF) | where |
|---|---|
| binary dataset, train/test split | FracAtlas (Kaggle `mahmudulhasantasin/fracatlas-original-dataset`); notebook §1, `src/data.py` |
| CNN + ScatNet, same classifier except input size | `src/models.py` (`Classifier`), notebook §2 (asserted) |
| k-fold CV: mean accuracy + mean F1 on the training set | `training.cross_validate`, notebook §3 |
| filters extracted and compared | notebook §6: `filters_cnn`, `filters_scatnet`, `frequency_coverage` |
| test set, >= 75% accuracy | notebook §4, `training.evaluate_test` |
| six XAI methods on CNN and ScatNet | `src/xai.py`: Saliency, Integrated Gradients, Guided Backprop, Grad-CAM, Occlusion, LIME |
| one method from scratch vs Captum | `src/occlusion_scratch.py`, notebook §8 |
| attributions overlaid, 2 images per class, both models | notebook §7 (`xai_cnn`, `xai_scatnet`) |
| quality of the attributions, methods that cannot be used | notebook §9 (deletion test, agreement), §10 (boxes), §12 |
| learning curves train + val in one figure | `learning_curves.png` |
| presentation | `presentation/main.pdf` (built from `results/`), `presentation/SPEAKER_NOTES.md` |

`python .claude/skills/exam-checklist/scripts/check_exam.py` checks all of this on a finished run.

## Results

The numbers are produced by the Kaggle run and written to `results/summary.json`
(`report.key_findings` prints them at the end of the notebook). They are not typed anywhere by hand:
the slides read `results/latex/`. After `./run.sh get`, the full table is in `results/latex/test.tex`
and in the notebook's §4-§5.

## Project layout

```
bone-fracture-detection/
├── notebooks/main.ipynb      the whole experiment: settings + calls to src (no function definitions)
├── src/
│   ├── bootstrap.py          Kaggle / Colab / local setup: packages, dataset path, output folders
│   ├── data.py               load + cache FracAtlas (X-rays + fracture boxes), near-duplicates, grouped split, loaders
│   ├── models.py             BoneFractureCNN, ScatNet, ResNet18, one shared Classifier
│   ├── training.py           grouped k-fold CV, final training, test metrics, bootstrap CI, McNemar
│   ├── xai.py                six XAI methods, applicability, deletion test, agreement, scratch vs Captum
│   ├── occlusion_scratch.py  Occlusion implemented from scratch
│   ├── localize.py           XAI heatmap -> fracture box; hit rate and IoU against the radiologists' boxes
│   ├── plots.py              every figure (fixed colours per model / class / method)
│   ├── report.py             summary.json + LaTeX macros/tables for the slides
│   └── utils.py              seed, device, parameter count, Kymatio/SciPy fix
├── tests/                    pytest: unit tests + the notebook run end to end on synthetic X-rays
├── presentation/             beamer slides (main.tex -> main.pdf), speaker notes
├── results/                  written by the notebook: summary.json, cv/, final/, figures/, latex/
│                             (models/ and attributions/ are large and gitignored)
├── docs/previous_run/        CV logs of the first iteration (before the fixes listed below)
├── kernel-metadata.json      Kaggle kernel: GPU, internet, dataset attached
├── run.sh                    Windows/Linux shell entry point
├── scripts/kaggle_run.py     private code snapshot + push / status / get
└── .claude/skills/           kaggle-run, exam-checklist, presentation, ml-project-structure
```

## Run it on Kaggle (recommended)

Same pattern as `macura-drone`: the job runs on a Kaggle GPU, launched from your terminal; you can
switch the PC off and download the results later.

**One-time setup**
1. Activate the environment and install the local dependencies:
   ```bash
   conda activate bone-fracture
   python -m pip install -r requirements.txt
   ```
2. Make sure the Kaggle account is phone-verified, then authenticate the CLI. OAuth is recommended
   because it requests the permissions needed to create private Datasets and update Notebooks. If a
   legacy key exists, move it aside first because it can take precedence over OAuth:
   ```bash
   mv ~/.kaggle/kaggle.json ~/.kaggle/kaggle.json.backup
   kaggle auth login --force
   ```
   Sign in with the same Kaggle account named in `kernel-metadata.json` and accept the requested
   permissions. If no browser opens, use
   `kaggle auth login --force --no-launch-browser` and open the displayed URL manually.
3. Put your Kaggle username in `kernel-metadata.json` (`"id": "<username>/bone-fracture-detection"`).

Do not put Kaggle keys or GitHub tokens in this repository. To restore the old legacy Kaggle key if
needed, run `mv ~/.kaggle/kaggle.json.backup ~/.kaggle/kaggle.json`.

### Setup for collaborators

After pulling the repository, each collaborator uses their own Kaggle account and credentials.
Change only the `id` line in `kernel-metadata.json`:

```json
"id": "COLLABORATOR_KAGGLE_USERNAME/bone-fracture-detection"
```

Then authenticate locally with that same account:

```bash
conda activate bone-fracture
python -m pip install -r requirements.txt
mv ~/.kaggle/kaggle.json ~/.kaggle/kaggle.json.backup  # only if the legacy file exists
kaggle auth login --force
./run.sh push --quick
```

No GitHub token is needed on Kaggle. Do not copy or commit another person's `kaggle.json`, access
token, OAuth credentials, or `kaggle_remember_setup.txt`. The launcher tests derive the expected
owner from `kernel-metadata.json`, so they do not require a username edit.

GitHub authentication is needed only for `git clone` / `git pull` on your computer. The remote
kernel never receives a GitHub token: `run.sh` uploads an allowlisted snapshot of `src/*.py` and
`requirements.txt` as an immutable **private Kaggle Dataset**, then attaches it to the notebook.
Identical source snapshots reuse that dataset. A changed snapshot creates one small private dataset
that can later be removed from Kaggle when its reproducibility record is no longer needed.

`kernel-metadata.json`: `code_file` = the notebook, `enable_gpu` with `machine_shape: NvidiaTeslaT4`
(recent PyTorch builds no longer support the older P100), `enable_internet` true (pip install
kymatio/captum + ResNet18 weights), `dataset_sources` =
`mahmudulhasantasin/fracatlas-original-dataset` (FracAtlas, mounted read-only under `/kaggle/input`).
The launcher adds the private code snapshot to `dataset_sources` only in its staged metadata.

**Run**
```bash
conda activate bone-fracture
cd ~/Desktop/bone-fracture-detection

./run.sh push --quick --dry-run  # inspect the staged files locally; no Kaggle connection
./run.sh push --quick     # 5-minute health check on a small subset (do it once)
./run.sh status           # queued / running / complete / error
./run.sh push             # the full run, ~1-2 h on a T4 (5-fold CV x 3 models x 20 epochs + final training + XAI)
./run.sh get              # download into ./out and copy the results into ./results
./run.sh slides           # rebuild presentation/main.pdf with the new numbers
./run.sh stop             # print the Kaggle page where a running session can be stopped
./run.sh                  # show launcher help
```
The launcher uploads the files from the currently checked-out local branch, including uncommitted
source edits. It records the branch, commit and snapshot checksum in the staged notebook metadata.
After Kaggle accepts the job, it runs independently and the PC can be switched off. Each download is
kept in a separate `out/run-<timestamp>/` folder so a failed run cannot erase an earlier result.
Outputs (in `/kaggle/working/results` on the kernel): `summary.json`, `cv/`, `final/`, `figures/`,
`latex/`, `models/*.pth`, `attributions/*.npz`.

Recommended order: run `push --quick`, monitor with `status`, and download with `get`. If that
health check succeeds, start the full run with `push`. Wait for `Kaggle accepted the job; the PC can
now be switched off.` before closing the terminal or turning off the computer.

If Dataset creation ends with `403 Forbidden` after the file upload reaches 100%, the file transfer
succeeded but Kaggle rejected creation of the Dataset. Repeat the OAuth setup above and accept the
Dataset/Notebook permissions; do not add a token to the notebook or repository.

## Run it locally

```bash
pip install -r requirements.txt
# dataset: download FracAtlas and unzip it into data/ (any depth: the folder with images/Fractured is found)
jupyter notebook notebooks/main.ipynb
```
Only the **Settings** cell needs editing (`QUICK`, `TRAIN`, `MODELS`, `EPOCHS`, ...). With `TRAIN = False`
the notebook reloads `results/models/*.pth` and the JSON logs of a previous run and only redraws.
Environment variables `BFD_DATA_DIR` / `BFD_RESULTS_DIR` override the paths without editing the notebook.

## Tests

```bash
pytest -m "not slow"   # ~2-5 min on a CPU: data, duplicates, models, training, every XAI method, report
pytest -m slow         # the whole notebook on synthetic X-rays (tests/synthetic.py), QUICK mode
```
What they prove without the real data: identical classifiers; every method runs on every model and
Grad-CAM is refused on ScatNet; our Occlusion equals Captum's (to 1e-5); Integrated Gradients
satisfies completeness; the deletion test ranks the correct map first; YOLO boxes are read and rescaled;
a map that is hot on a (synthetic) fracture gives a box on it; the split and the folds never separate a
duplicate group; planted flipped/rotated/brightened copies are found; a model overfits 16 images;
the notebook runs top to bottom. CI runs both on every push (`.github/workflows/tests.yml`).

## Design choices (and why)

- **Grey input, 1 channel**: X-rays are grey; 3 identical channels only triple ScatNet's coefficients.
- **CNN**: 4 conv blocks 32 -> 256 with BN and max-pool; the **first layer is 7x7** so that its filters
  show a shape that can be compared with the wavelets.
- **ScatNet**: Kymatio `Scattering2D(J=4, L=8)`, second order: 417 maps on the same 14x14 grid as the
  CNN; coefficients log-compressed and batch-normalised (their magnitudes span orders).
- **Same classifier** `flatten -> 512 -> 128 -> 2` (ReLU, dropout 0.5) for every model.
- **FracAtlas**: the only public fracture dataset of this size with fracture boxes drawn by radiologists,
  so the explanations can be checked, not only looked at. It has no official split: we make a stratified
  ~70/15/15 split.
- **Near-duplicates**: copies of the same X-ray (resized, flipped, brighter) are found first, and the
  split and the cross-validation folds are **grouped**: copies never end up on both sides.
- **Unbalanced classes**: 1 X-ray in 6 is fractured, so the cross-entropy weights each class by its
  inverse frequency, the best model is chosen by F1 of the fractured class, and recall and AUC are
  reported next to accuracy (always answering "not fractured" would already give ~82%).
- **Honest CV**: fold scores at the last epoch (nothing selected on the validation fold); the best model
  is chosen on CV (`SELECT_BY = "f1"`), never on the test set; the test set is used once, with a
  bootstrap 95% interval and McNemar tests between models.
- **Final models** trained on the whole training split; the separate `val` split picks the epoch.
- **XAI**: six methods from four families, so the discussion can compare them and show that
  **Grad-CAM cannot be applied to ScatNet** (no learned conv feature map) and **Guided Backprop only
  partly** (the scattering non-linearity is a modulus, not a ReLU). "Removed" pixels are **black**
  (the X-ray background): a grey baseline made the first version highlight the background.
- **Quality of the attributions** is measured, not only looked at: deletion test (faithfulness) and
  Spearman agreement between methods.
- **Fracture boxes** (`src/localize.py`): each map is smoothed, the pixels above half of the strongest
  evidence are kept, and the box goes around the connected area with the most evidence. Scores on the
  fractured test X-rays: **hit rate** (pointing game: hottest point inside a radiologist's box, against a
  random point) and **IoU** of the boxes. The figures keep the evidence outside the box visible (faint),
  so a model that also looks at a label or a metal plate is not hidden.

## Previous iteration and what changed

The first version (branch `xai`, team notebook) used the Kaggle *Bone Fracture Multi-Region X-ray
Data* and reached CV 97.4 / 98.3 / 96.7% and test 92.9 / 94.7 / 90.5% for CNN / ResNet18 / ScatNet
(logs in `docs/previous_run/`; not comparable with FracAtlas). It had: different classifiers for
CNN and ScatNet, CV folds mixing copies of an X-ray and scored at the best epoch (4-6 points CV/test
gap), RGB input, a grey Occlusion baseline, F1 computed for the wrong class, no Kaggle setup, and a
dataset without fracture locations, so the explanations could not be checked. All fixed here. The old notebooks and attribution files are in the history of branch `xai`.

> **Security**: a Kaggle API key was committed in an old notebook (branch `xai` history). Revoke it
> on kaggle.com (Settings -> API -> Expire token) if not done yet.

## References

Bruna & Mallat 2013 (scattering networks) · Simonyan et al. 2014 (saliency) · Springenberg et al.
2015 (guided backprop) · Zeiler & Fergus 2014 (occlusion) · Ribeiro et al. 2016 (LIME) · Sundararajan
et al. 2017 (integrated gradients) · Selvaraju et al. 2017 (Grad-CAM) · Samek et al. 2017 (deletion /
region perturbation) · Zhang et al. 2018 (pointing game) · Abedeen et al. 2023 (FracAtlas, *Scientific
Data*) · Captum (captum.ai) · Kymatio (kymat.io).

Course instructors: Prof. Gloria Menegaz, Giorgio Dolci.
