# Bone Fracture Detection: CNN vs ScatNet + Explainable AI

MSc in Artificial Intelligence, Visual Intelligence 2025/2026, University of Verona

Binary classification of bone X-rays (**fractured** / **not fractured**) with a custom CNN, a
wavelet Scattering Network (ScatNet) and a pretrained ResNet18 baseline, explained with six XAI
methods; Occlusion is also implemented from scratch and compared with Captum.

## Structure

```
bone-fracture-detection/
├── notebooks/main.ipynb       the whole experiment: settings + calls to src, no function definitions
├── src/
│   ├── data.py                dataset, transforms, loaders, dataset summary
│   ├── models.py              BoneFractureCNN, ScatNet2D, BoneFractureResNet18, build_model()
│   ├── training.py            5-fold CV, test metrics, save/load weights and results
│   ├── xai.py                 the six Captum methods, caching, scratch-vs-Captum, method agreement
│   ├── occlusion_scratch.py   Occlusion implemented from scratch (the "custom" method)
│   ├── plots.py               every figure of the notebook
│   └── utils.py               seed, device, parameter count
├── results/
│   ├── cv/<model>.json        CV metrics + learning curves (current files: imported from the last full run)
│   ├── models/<model>.pth     trained weights (not in git: copy them here, see below)
│   ├── attributions/          cached XAI maps (.npz), created by the notebook
│   └── figures/               figures for the report, created by the notebook
├── archive/                   everything from before the refactor (old notebooks, old src, old attribution files)
├── .claude/skills/ml-project-structure/   Claude Code skill that builds/keeps this structure
└── requirements.txt
```

Why this layout:
- **One home for every function.** Before, the same plotting code existed in the notebook and in
  `src/`, there were three saliency functions and several unused modules. Now each step of the
  pipeline is one flat module, imported as `from src import data, models, training, xai, plots`.
- **The notebook reads like the report.** One settings cell, then one section per exam step with
  short cells (1 to 9 lines) that call `src`. It went from 190 cells / 51 MB to 27 cells.
- **Expensive steps are optional.** `TRAIN = False` reloads saved weights and CV logs; XAI maps are
  cached, so re-running the notebook for figures takes minutes, not hours.
- **Nothing was deleted.** Old material is in `archive/` (and in git history); delete that folder
  whenever you no longer need it.

## How to run

```bash
pip install -r requirements.txt
```

1. Download the Kaggle dataset *Bone Fracture Multi-Region X-ray Data* into
   `Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/` (with `train/`,
   `val/`, `test/`), plus the `choosen_test/` folder with the 8 images used for XAI.
2. Open `notebooks/main.ipynb` and edit only the **Settings** cell.
3. `TRAIN = True` runs the 5-fold CV for every model and saves `results/models/<model>.pth`.
   `TRAIN = False` needs those files. Weights trained before the refactor load unchanged, just rename them:
   `custom_cnn_best_fold_model.pth → cnn.pth`, `resnet18_best_fold_model.pth → resnet18.pth`,
   `scatnet2D_best_fold_model.pth → scatnet.pth`.

On **Colab**: copy the repository to `MyDrive/bone-fracture-detection/`, open the notebook and run;
the setup cell mounts Drive and installs `kymatio` and `captum`. Set `DATA_DIR` to where the data is.

## Results (last full run, before the refactor)

| model | CV accuracy (5-fold) | test accuracy | test F1 (fractured) | test recall (fractured) | parameters |
|---|---|---|---|---|---|
| Custom CNN | 97.4 ± 0.5 % | 92.9 % | 92.4 % | 92.4 % | 26.1 M |
| ResNet18 (pretrained) | 98.3 ± 0.7 % | 94.7 % | 94.4 % | 95.0 % | 11.4 M |
| ScatNet (J=4, L=8) | 96.7 ± 0.5 % | 90.5 % | 90.2 % | 92.4 % | 125.5 M |

Test set: 506 images (238 fractured, 268 not fractured). All models exceed the 75 % target.
An earlier ScatNet with global average pooling of the coefficients reached only 80.2 % test
accuracy, most likely because pooling throws away *where* in the image each wavelet responds.

## Known issues / next steps

1. **Classifier heads differ (exam requirement).** The exam asks for the same final classifier
   in CNN and ScatNet except for the number of input neurons. The CNN head is
   `Linear(→512) → ReLU → Dropout → Linear(512→128) → ReLU → Dropout → Linear(128→2)`, the ScatNet
   head is `Linear(→512) → BatchNorm → ReLU → Dropout → Linear(512→2)`. Fixing it means
   retraining one of the two models.
2. **CV is optimistic.** CV accuracy is 3.7 to 6.2 points above test accuracy for every model. The training
   folder may contain several (augmented) versions of the same X-ray, which then end up on
   both sides of a CV split. A duplicate check (image hashes) would confirm it.
3. **Occlusion baseline.** With `baseline=0.0` (grey) Occlusion highlights the black background,
   because a grey square on black is an unrealistic input. `XAI_PARAMS["baseline"] = -1.0` (black)
   is the better choice for X-rays; it changes IG, LIME and Shapley too.
4. **Behaviour changes made in the refactor** (re-run with `TRAIN = True` to get consistent numbers):
   - CV validation folds are no longer evaluated with training augmentation, folds are stratified;
   - the saved model is the best fold (before: always the last fold);
   - F1/precision/recall are computed for *fractured* (before: *not fractured*);
   - every XAI method explains the true class (before: LIME/Shapley used the predicted class);
   - the from-scratch Occlusion now averages overlapping windows like Captum, so the two match
     exactly (tested on a random CNN: max difference 5e-8; before: r = 0.97);
   - confusion-matrix labels come from the dataset (before: 3 of 4 plots had swapped labels);
   - the ScatNet saliency is computed with ScatNet (before: accidentally with the CNN).
5. **Revoke the Kaggle API key** that was committed in the old notebook (redacted in `archive/`,
   but still present in git history).
6. **Not yet verified end to end:** the new notebook has not been run on the real data. Run it
   once with `TRAIN = False` (or `True`) before submitting.

## References

Captum (captum.ai), Kymatio (kymat.io); Bruna & Mallat 2013 (scattering networks), Sundararajan
et al. 2017 (Integrated Gradients), Ribeiro et al. 2016 (LIME), Zeiler & Fergus 2014 (Occlusion).

Course instructors: Prof. Gloria Menegaz, Giorgio Dolci.
