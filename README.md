# Bone Fracture Detection: CNN vs ScatNet, XAI, fracture boxes and a YOLO pipeline

MSc in Artificial Intelligence, Visual Intelligence 2025/2026, University of Verona

X-rays of **FracAtlas** (fractured / not fractured, with the radiologists' fracture boxes). A CNN, a wavelet
ScatNet and a pretrained ResNet18, all ending with **the same classifier**, are compared and explained with
**six XAI methods** (Occlusion also written from scratch). The explanations become **fracture boxes**,
checked against the radiologists' boxes, and a **full pipeline** after Linda (2025) combines the best
classifier, a **YOLOv8** detector and the best XAI method into a short report per X-ray.

## Run it on Kaggle (macOS and Windows)

### Once, on each computer

1. **Python 3.9+** and the Kaggle tool: `python -m pip install kaggle`
2. A **phone-verified** Kaggle account (needed for the GPU and internet).
3. A **Kaggle API token**: kaggle.com > Settings > API > *Create New Token*. Put the downloaded `kaggle.json` in
   - macOS / Linux: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<you>\.kaggle\kaggle.json`

   The notebook runs on **this** account. Check which one with `kaggle config view` (the username line).

### Every run

| | macOS, Linux, Windows Git Bash | Windows PowerShell or cmd |
|---|---|---|
| quick check (~5 min) | `./run.sh push --quick` | `run.bat push --quick` |
| full run (~2-3 h on a T4) | `./run.sh push` | `run.bat push` |
| is it done? | `./run.sh status` | `run.bat status` |
| download the results | `./run.sh get` | `run.bat get` |

- `push` packs `src/` into the notebook and starts it at `kaggle.com/code/<your-username>/bone-fracture-detection`.
  When it prints *Kaggle accepted the job*, the computer can be switched off.
- `get` downloads everything into `out/run-<time>/` and, if the run is complete, copies `results/` into `./results`.
- Do the quick check first; start the full run only when it ends with status *complete*.

### If something goes wrong

| message | fix |
|---|---|
| *Kaggle login failed* | put a fresh `kaggle.json` in the `.kaggle` folder (step 3) |
| *Kaggle refused the request* (403) | the account is not phone-verified, or `--user` is not the logged-in account |
| *Kaggle username not found* | `./run.sh push --user <your-kaggle-username>` |
| status *error* | `./run.sh get`, then read the `.log` file in `out/run-<time>/` |
| *Python 3.9+ not found* | install Python (python.org) or activate your conda environment |

Two Kaggle accounts on one computer: the one used is the `kaggle.json` in the `.kaggle` folder.

## Results and slides

`results/summary.json` holds every number, `results/figures/` every figure, `results/latex/` the numbers of the
slides. `make -C presentation` (or `./run.sh slides`) rebuilds `presentation/main.pdf` from them;
`presentation/SPEAKER_NOTES.md` has the timing (about 11:30 of the 12 minutes).

## Run locally (optional)

```bash
python -m pip install -r requirements.txt
# unzip FracAtlas into data/ (the folder with images/Fractured is found at any depth)
jupyter notebook notebooks/main.ipynb
```

Only the **Settings** cell is edited (`QUICK`, `TRAIN`, `EPOCHS`, ...). `TRAIN = False` reloads the weights of a
previous run (`results/models/`, `results/detector/`) and only redraws.

## Tests

```bash
pytest -m "not slow"   # a few minutes on a CPU, synthetic X-rays
pytest -m slow         # the whole notebook on synthetic X-rays (needs ultralytics)
```

## Project layout

```
notebooks/main.ipynb   the experiment: settings + calls to src/
src/
  bootstrap.py         Kaggle or local setup
  data.py              FracAtlas, fracture boxes, near-duplicates, split, loaders
  models.py            CNN, ScatNet, ResNet18, one shared Classifier
  training.py          grouped k-fold CV, final training, test metrics
  xai.py               six XAI methods, deletion test, agreement
  occlusion_scratch.py Occlusion from scratch
  localize.py          heatmap -> fracture box, hit rate and IoU
  detect.py            YOLOv8 on the fracture boxes
  pipeline.py          classifier + YOLO + XAI -> verdict and report
  plots.py, report.py  figures, summary.json and the numbers of the slides
tests/                 pytest (synthetic X-rays)
presentation/          slides (main.tex -> main.pdf) and speaker notes
run.sh, run.bat        Kaggle launcher (scripts/kaggle_run.py)
kernel-metadata.json   Kaggle settings: T4 GPU, internet, FracAtlas attached
```

## Method in short

- **Data**: FracAtlas (4,083 X-rays, 717 fractured). Grey, 224x224. Copies of the same X-ray are found first,
  so the stratified ~70/15/15 split and the k folds never separate them. 1 in 6 is fractured: weighted loss.
- **Models**: CNN (first layer 7x7), ScatNet (Kymatio, J=4, L=8), ResNet18; same classifier
  `flatten -> 512 -> 128 -> 2`. Best model by cross-validation F1, test set used once.
- **XAI**: Saliency, Integrated Gradients, Guided Backprop, Grad-CAM (not on ScatNet), Occlusion, LIME;
  black = "removed". Quality: deletion test and hit rate against the radiologists' boxes.
- **Pipeline** (Linda 2025, without CT and the graph network): YOLOv8s gives the box, the best classifier
  decides, the best XAI method explains; if classifier and YOLO disagree, the X-ray *needs review*.

## Exam requirements -> where

| requirement | where |
|---|---|
| CNN + ScatNet, same classifier | `src/models.py`, notebook §2 |
| k-fold CV, mean accuracy + F1 | `training.cross_validate`, §3 |
| test set, >= 75% accuracy | `training.evaluate_test`, §4 |
| filters compared | §6 |
| six XAI methods, 2 images per class | `src/xai.py`, §7 |
| one method from scratch vs Captum | `src/occlusion_scratch.py`, §8 |
| quality of the explanations, discussion | §9, §10, §13 |

`python .claude/skills/exam-checklist/scripts/check_exam.py` checks a finished run.

## Previous version

The first version used the Kaggle *Bone Fracture Multi-Region* dataset: 98.6% of its test X-rays had a copy in
the training set, so its 95-99% test accuracy mostly measured memory, and it had no fracture boxes. Its logs are
in `docs/previous_run/`. (An old Kaggle key is in the history of branch `xai`: revoke it on kaggle.com.)

## References

Bruna & Mallat 2013 (scattering) · Simonyan et al. 2014 (saliency) · Springenberg et al. 2015 (guided backprop)
· Zeiler & Fergus 2014 (occlusion) · Ribeiro et al. 2016 (LIME) · Sundararajan et al. 2017 (integrated
gradients) · Selvaraju et al. 2017 (Grad-CAM) · Samek et al. 2017 (deletion test) · Zhang et al. 2018 (pointing
game) · Abedeen et al. 2023 (FracAtlas, *Scientific Data*) · Linda 2025 (fracture detection and reporting,
*J. Electrical Systems*) · Captum · Kymatio · Ultralytics YOLOv8.
