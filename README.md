# Bone Fracture Detection: CNN vs ScatNet, explained with XAI

MSc in Artificial Intelligence, Visual Intelligence 2025/2026, University of Verona

Binary classification of X-rays (**fractured** / **not fractured**). A CNN trained from scratch and a
wavelet Scattering Network (ScatNet, Kymatio) are compared, with an ImageNet-pretrained ResNet18 as a
reference; all three end with **the same classifier**. The best model is chosen by cross-validation,
tested once, its filters are compared with ScatNet's wavelets, and the models are explained with
**six XAI methods** (Captum), one of which (Occlusion) is also implemented from scratch.

Everything runs from one notebook, `notebooks/main.ipynb`, on a **Kaggle GPU** (`./run.sh push`).

## Exam requirements -> where

| requirement (exam PDF) | where |
|---|---|
| binary dataset, train/test split | Kaggle *Bone Fracture Multi-Region X-ray Data*; notebook §1, `src/data.py` |
| CNN + ScatNet, same classifier except input size | `src/models.py` (`Classifier`), notebook §2 (asserted) |
| k-fold CV: mean accuracy + mean F1 on the training set | `training.cross_validate`, notebook §3 |
| filters extracted and compared | notebook §6: `filters_cnn`, `filters_scatnet`, `frequency_coverage` |
| test set, >= 75% accuracy | notebook §4, `training.evaluate_test` |
| six XAI methods on CNN and ScatNet | `src/xai.py`: Saliency, Integrated Gradients, Guided Backprop, Grad-CAM, Occlusion, LIME |
| one method from scratch vs Captum | `src/occlusion_scratch.py`, notebook §8 |
| attributions overlaid, 2 images per class, both models | notebook §7 (`xai_cnn`, `xai_scatnet`) |
| quality of the attributions, methods that cannot be used | notebook §9 (deletion test, agreement), §11 |
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
│   ├── data.py               load + cache X-rays, dataset study (sizes, balance, near-duplicates), loaders
│   ├── models.py             BoneFractureCNN, ScatNet, ResNet18, one shared Classifier
│   ├── training.py           grouped k-fold CV, final training, test metrics, bootstrap CI, McNemar
│   ├── xai.py                six XAI methods, applicability, deletion test, agreement, scratch vs Captum
│   ├── occlusion_scratch.py  Occlusion implemented from scratch
│   ├── plots.py              every figure (fixed colours per model / class / method)
│   ├── report.py             summary.json + LaTeX macros/tables for the slides
│   └── utils.py              seed, device, parameter count, Kymatio/SciPy fix
├── tests/                    pytest: unit tests + the notebook run end to end on synthetic X-rays
├── presentation/             beamer slides (main.tex -> main.pdf), speaker notes
├── results/                  written by the notebook: summary.json, cv/, final/, figures/, latex/
│                             (models/ and attributions/ are large and gitignored)
├── docs/previous_run/        CV logs of the first iteration (before the fixes listed below)
├── kernel-metadata.json      Kaggle kernel: GPU, internet, dataset attached
├── run.sh                    push / status / get / slides / stop
└── .claude/skills/           kaggle-run, exam-checklist, presentation, ml-project-structure
```

## Run it on Kaggle (recommended)

Same pattern as `macura-drone`: the job runs on a Kaggle GPU, launched from your terminal; you can
switch the PC off and download the results later.

**One-time setup**
1. `pip install kaggle`; on kaggle.com: Settings -> *Create New Token*, save it as `~/.kaggle/kaggle.json`
   (`chmod 600`; Windows: `C:\Users\<you>\.kaggle\kaggle.json`).
2. On kaggle.com add a **Secret** `GITHUB_TOKEN`: a GitHub token that can read this private repo
   (the kernel clones the code with it).
3. Put your Kaggle username in `kernel-metadata.json` (`"id": "<username>/bone-fracture-detection"`).

`kernel-metadata.json`: `code_file` = the notebook, `enable_gpu` and `enable_internet` true (clone +
pip install kymatio/captum + ResNet18 weights), `dataset_sources` =
`bmadushanirodrigo/fracture-multi-region-x-ray-data` (mounted read-only under `/kaggle/input`).

**Run**
```bash
./run.sh push --quick     # 5-minute health check on a small subset (do it once)
./run.sh status           # queued / running / complete / error
./run.sh push             # the full run, ~3-4 h on a P100/T4 (5-fold CV x 3 models + final training + XAI)
./run.sh get              # download into ./out and copy the results into ./results
./run.sh slides           # rebuild presentation/main.pdf with the new numbers
```
The kernel clones branch `main`; for another branch: `BRANCH=<branch> ./run.sh push`.
Outputs (in `/kaggle/working/results` on the kernel): `summary.json`, `cv/`, `final/`, `figures/`,
`latex/`, `models/*.pth`, `attributions/*.npz`.

## Run it locally

```bash
pip install -r requirements.txt
# dataset: download the Kaggle dataset and unzip it into data/ (any depth: the train/ val/ test/ folder is found)
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
satisfies completeness; the deletion test ranks the correct map first; grouped folds never split a
duplicate group; planted flipped/rotated/brightened copies are found; a model overfits 16 images;
the notebook runs top to bottom. CI runs both on every push (`.github/workflows/tests.yml`).

## Design choices (and why)

- **Grey input, 1 channel**: X-rays are grey; 3 identical channels only triple ScatNet's coefficients.
- **CNN**: 4 conv blocks 32 -> 256 with BN and max-pool; the **first layer is 7x7** so that its filters
  show a shape that can be compared with the wavelets.
- **ScatNet**: Kymatio `Scattering2D(J=4, L=8)`, second order: 417 maps on the same 14x14 grid as the
  CNN; coefficients log-compressed and batch-normalised (their magnitudes span orders).
- **Same classifier** `flatten -> 512 -> 128 -> 2` (ReLU, dropout 0.5) for every model.
- **Near-duplicates**: the training set contains copies of the same X-ray. Cross-validation is
  **grouped** by copy, and test accuracy is also reported on the **clean** test images (no copy in train).
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

## Previous iteration and what changed

The first version (branch `xai`, team notebook) reached CV 97.4 / 98.3 / 96.7% and test 92.9 / 94.7 /
90.5% for CNN / ResNet18 / ScatNet (logs in `docs/previous_run/`). It had: different classifiers for
CNN and ScatNet, CV folds mixing copies of an X-ray and scored at the best epoch (4-6 points CV/test
gap), RGB input, a grey Occlusion baseline, F1 computed for the wrong class, and no Kaggle setup.
All fixed here. The old notebooks and attribution files are in the history of branch `xai`.

> **Security**: a Kaggle API key was committed in an old notebook (branch `xai` history). Revoke it
> on kaggle.com (Settings -> API -> Expire token) if not done yet.

## References

Bruna & Mallat 2013 (scattering networks) · Simonyan et al. 2014 (saliency) · Springenberg et al.
2015 (guided backprop) · Zeiler & Fergus 2014 (occlusion) · Ribeiro et al. 2016 (LIME) · Sundararajan
et al. 2017 (integrated gradients) · Selvaraju et al. 2017 (Grad-CAM) · Samek et al. 2017 (deletion /
region perturbation) · Captum (captum.ai) · Kymatio (kymat.io).

Course instructors: Prof. Gloria Menegaz, Giorgio Dolci.
