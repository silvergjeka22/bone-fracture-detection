"""Bone fracture detection: CNN vs ScatNet (+ ResNet18) with explainable AI.

Modules, in the order the notebook (notebooks/main.ipynb) uses them:
    bootstrap          Kaggle / Colab / local setup: packages, dataset path, output folders
    data               load + cache the X-rays, dataset study (sizes, balance, near-duplicates), loaders
    models             BoneFractureCNN, ScatNet, ResNet18, all ending in the same Classifier
    training           group-aware k-fold CV, final training, test metrics, McNemar, tables
    xai                six XAI methods (Captum), deletion test, method agreement
    occlusion_scratch  Occlusion implemented from scratch (compared against Captum)
    plots              every figure
    report             summary.json + LaTeX numbers/tables for the presentation
    utils              seed, device, parameter count
"""
