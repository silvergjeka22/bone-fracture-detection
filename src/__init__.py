"""Bone fracture detection: CNN vs ScatNet (+ ResNet18) with explainable AI.

Modules, in the order the notebook (notebooks/main.ipynb) uses them:
    bootstrap          Kaggle or local setup: packages, dataset path, output folders
    data               load + cache FracAtlas (X-rays + fracture boxes), near-duplicates, grouped split, loaders
    models             BoneFractureCNN, ScatNet, ResNet18, JointNet, all ending in the same Classifier
    training           group-aware k-fold CV, final training, test metrics, McNemar, tables
    xai                six XAI methods (Captum), deletion test, method agreement
    occlusion_scratch  Occlusion implemented from scratch (compared against Captum)
    localize           XAI heatmap -> fracture box, scored against the radiologists' boxes
    joint              our method: classifier + built-in detector + XAI losses, trained together
    detect             YOLOv8 trained on the fracture boxes (the detector of the full pipeline)
    pipeline           classifier + YOLO + explanation -> verdict and report per X-ray (after Linda, 2025)
    plots              every figure
    report             results/summary.json and the key findings
    utils              seed, device, parameter count
"""
