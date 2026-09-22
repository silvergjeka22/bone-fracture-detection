"""Bone fracture detection: CNN vs ScatNet with explainable AI.

Modules (one per step of the pipeline, used in this order by notebooks/main.ipynb):
    data              datasets, transforms, loaders
    models            BoneFractureCNN, BoneFractureResNet18, ScatNet2D
    training          k-fold cross-validation, test evaluation, saving/loading weights
    xai               the six Captum XAI methods + caching + method agreement
    occlusion_scratch Occlusion implemented from scratch (compared against Captum)
    plots             every figure of the notebook
    utils             seed, device, parameter count
"""
