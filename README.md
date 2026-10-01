# Bone Fracture Detection: CNN vs ScatNet, XAI, guided training and a YOLO pipeline

MSc in Artificial Intelligence, Visual Intelligence 2025/2026, University of Verona

X-rays of **FracAtlas** (fractured / not fractured, with the radiologists' fracture boxes). A CNN, a wavelet
ScatNet and a pretrained ResNet18, all ending with **the same classifier**, are compared and explained with
**six XAI methods** (Occlusion also written from scratch). The explanations become **fracture boxes**,
checked against the radiologists' boxes. The best model is then **trained again to look at the fracture**
(box loss + contrastive loss), and a **full pipeline** after Linda (2025) combines the classifier, a **YOLOv8**
detector and the best XAI method into a short report per X-ray.

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
| full run (~3 h on a T4) | `./run.sh push` | `run.bat push` |
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

## Results

`results/summary.json` holds every number and `results/figures/` every figure; the end of the notebook prints
the key findings.

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
  guidance.py          guided training: box loss + contrastive loss
  detect.py            YOLOv8 on the fracture boxes
  pipeline.py          classifier + YOLO + XAI -> verdict and report
  plots.py, report.py  figures, summary.json and the key findings
tests/                 pytest (synthetic X-rays)
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
- **Guided training**: the best model is trained again with
  `loss = weighted CE + box x Energy loss + contrast x contrastive loss` on the fractured training X-rays:
  the Energy loss (Rao et al. 2023) is the share of its Grad-CAM heat outside the radiologist's box; the
  contrastive loss (SupCon, SimCLR-style) makes fracture regions alike and background different. Warm-up first,
  boxes grown a little, Grad-CAM on the 14x14 layer. Compared with the plain model: accuracy, F1, hit rate of
  all six XAI methods, deletion test.
- **Pipeline** (Linda 2025, without CT and the graph network): YOLOv8s (trained separately on the boxes) gives
  the box, the classifier decides, the XAI method explains; model and method are chosen on the val split.
  If classifier and YOLO disagree, the X-ray *needs review*.

## Exam requirements -> where

| requirement | where |
|---|---|
| CNN + ScatNet, same classifier | `src/models.py`, notebook §2 |
| k-fold CV, mean accuracy + F1 | `training.cross_validate`, §3 |
| test set, >= 75% accuracy | `training.evaluate_test`, §4 |
| filters compared | §6 |
| six XAI methods, 2 images per class | `src/xai.py`, §7 |
| one method from scratch vs Captum | `src/occlusion_scratch.py`, §8 |
| quality of the explanations | §9, §10, §11 |

`python .claude/skills/exam-checklist/scripts/check_exam.py` checks a finished run.

## Previous version

The first version used the Kaggle *Bone Fracture Multi-Region* dataset: 98.6% of its test X-rays had a copy in
the training set, so its 95-99% test accuracy mostly measured memory, and it had no fracture boxes. Its logs are
in `docs/previous_run/`. (An old Kaggle key is in the history of branch `xai`: revoke it on kaggle.com.)

## References

Bruna & Mallat 2013 (scattering) · Simonyan et al. 2014 (saliency) · Springenberg et al. 2015 (guided backprop)
· Zeiler & Fergus 2014 (occlusion) · Ribeiro et al. 2016 (LIME) · Sundararajan et al. 2017 (integrated
gradients) · Selvaraju et al. 2017 (Grad-CAM) · Samek et al. 2017 (deletion test) · Zhang et al. 2018 (pointing
game) · Ross et al. 2017 (right for the right reasons) · Li et al. 2018 (GAIN) · Rao et al. 2023 (model
guidance with boxes, Energy loss) · Chen et al. 2020 (SimCLR) · Khosla et al. 2020 (supervised contrastive) ·
Selvaraju et al. 2021 (CAST) · Abedeen et al. 2023 (FracAtlas, *Scientific Data*) · Linda 2025 (fracture detection
and reporting, *J. Electrical Systems*) · Captum · Kymatio · Ultralytics YOLOv8.
