# Worked example: bone-fracture-detection (Visual Intelligence exam, 2026)

## Before

- `notebooks/bone_fracture_detection_cnn.ipynb`: 190 cells, 51 MB, plus a 46 MB
  `... copy.ipynb` that was actually the newest version. 6 visualisation functions were
  defined inside the notebook and copy-pasted again into `src/visualization/xai_attributes.py`.
- `src/` had 6 sub-packages and 26 files: 3 saliency functions, 2 Input x Gradient functions,
  `visualize_triplet` defined twice, unused modules (GradCAM, DeepLIFT, Guided Backprop,
  Optuna search, VGG/ResNet50) and a README describing methods the notebook never used.
- 28 attribution files without extension saved next to the notebook.
- A Kaggle API key in a notebook cell, a Windows absolute path in the setup cell.

Bugs found by reading the outputs (not by running anything):
- three confusion matrices had swapped axis labels (label 0 was 'fractured', plotted as 'No Fracture');
- the "ScatNet saliency" maps were computed with the CNN;
- LIME/Shapley silently explained the predicted class, the other methods the true class;
- CV validation folds were evaluated with training augmentation;
- the file saved as "best fold model" was the last fold;
- F1 was computed for the 'not fractured' class;
- the from-scratch Occlusion overwrote overlapping windows instead of averaging (r = 0.97 vs Captum).

## After

```
src/  __init__.py data.py models.py training.py xai.py occlusion_scratch.py plots.py utils.py
notebooks/main.ipynb        27 cells, 17 short code cells, 15 KB
results/cv/*.json           CV logs parsed from the old notebook outputs
archive/                    old notebooks (key redacted), old src/, old attribution files
```

Notebook sections: Setup, Settings, 1 Data, 2 Models, 3 Cross-validation, 4 Test,
5 Filters, 6 XAI, 7 Scratch vs library, 8 Method agreement, 9 Discussion.

Typical cells:

```python
cv = {}
for name in MODELS:
    if TRAIN:
        cv[name] = training.train_kfold_cv(name, datasets["train"], datasets["train_eval"], device, ...)
    else:
        cv[name] = training.load_cv_results(name, RESULTS_DIR)
training.cv_table(cv)
```

```python
plots.plot_confusion_matrices(test, class_names, save_to=FIG_DIR / "confusion_matrices.png")
```

Verification that worked without the dataset or a GPU: the from-scratch Occlusion was
compared with Captum on a random CNN (max difference 5e-8), and the whole notebook was run
on 24 random images with 1 epoch and tiny XAI parameters.
