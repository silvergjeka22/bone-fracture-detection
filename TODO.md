# TODO — Bone Fracture Detection (Visual Intelligence Exam)

**University of Verona — MSc AI — 2025/2026**

This file tracks every requirement from the exam PDF against the current codebase state.

---

## STATUS LEGEND
- ✅ Done
- ⚠️ Partial / needs fix
- ❌ Missing / not implemented

---

## PART 1 — DATA & SETUP

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1.1 | Binary classification dataset selected (Fractured vs Not-Fractured X-rays) | ✅ | FracAtlas / Kaggle bone fracture dataset |
| 1.2 | `BoneFractureDataset` loads images from folder structure | ✅ | `src/data/datasets.py` |
| 1.3 | Train / Test split | ⚠️ | Dataset split logic must be explicit in notebook; confirm 80/20 |
| 1.4 | Data augmentation (required for good filter shapes) | ❌ | Add RandomHorizontalFlip, RandomRotation, ColorJitter to train transforms |
| 1.5 | Normalization with ImageNet mean/std | ⚠️ | Must be in transforms, not just resize |

---

## PART 2 — CNN MODEL

| # | Task | Status | Notes |
|---|------|--------|-------|
| 2.1 | `BoneFractureCNN` 4-block architecture (32→64→128→256) | ✅ | `src/models/cnn.py` |
| 2.2 | BatchNorm + MaxPool + Dropout | ✅ | |
| 2.3 | Fully connected classifier (fc1→fc2→fc3) | ✅ | |
| 2.4 | `get_conv1_filters()` method | ✅ | Returns first-layer weights |
| 2.5 | Forward pass returns logits (not softmax) | ✅ | Required for Captum XAI |

---

## PART 3 — SCATNET MODEL ❌ CRITICAL MISSING

| # | Task | Status | Notes |
|---|------|--------|-------|
| 3.1 | `BoneFractureScatNet` using Kymatio `Scattering2D` | ❌ | File raises `NotImplementedError` |
| 3.2 | J=2, L=8 scattering transform | ❌ | |
| 3.3 | Flatten scattering coefficients → FC classifier | ❌ | |
| 3.4 | **Same classifier head as CNN** (fc1→fc2→fc3, only input size differs) | ❌ | Exam requirement: same final classifier |
| 3.5 | `get_scatnet_filters()` to visualize wavelets | ❌ | Extract Morlet wavelet filters from Kymatio |
| 3.6 | ScatNet works in both `train` and `eval` mode | ❌ | |

**Implementation note:**
```python
from kymatio.torch import Scattering2D
# Input: (B, 1, H, W) — grayscale or convert to 1ch
# J=2, shape=(224,224) → output: (B, C_scat, H', W')
# Flatten → same FC head as CNN (different in_features)
```

---

## PART 4 — TRAINING & CROSS-VALIDATION

| # | Task | Status | Notes |
|---|------|--------|-------|
| 4.1 | `train_epoch` / `validate_epoch` functions | ✅ | `src/training/trainer.py` |
| 4.2 | K-Fold CV (k=5) on training set | ✅ | `src/training/cross_validation.py` |
| 4.3 | Mean accuracy across folds | ✅ | |
| 4.4 | **Mean F1 score across folds** | ❌ | `cross_validation.py` never computes F1 — EXAM REQUIRES IT |
| 4.5 | CV works for ScatNet (different model signature) | ❌ | `train_kfold_cv` hardcodes `dropout_rate` param — ScatNet has different `__init__` |
| 4.6 | Learning curves (train+val loss and acc in same plot) | ⚠️ | Plot function exists but not called with both curves in same figure |
| 4.7 | Early stopping or scheduler | ✅ | ReduceLROnPlateau used |
| 4.8 | Test set evaluation (accuracy + F1 + confusion matrix) | ⚠️ | `evaluator.py` exists but not verified |

**Fix needed in `cross_validation.py`:**
```python
from sklearn.metrics import f1_score
# After validate_epoch, compute f1_score(all_labels, all_preds, average='binary')
# Store in fold_results and report mean F1 at the end
```

---

## PART 5 — FILTER VISUALIZATION

| # | Task | Status | Notes |
|---|------|--------|-------|
| 5.1 | Visualize CNN conv1 filters (32 filters, 3 channels each) | ❌ | `get_conv1_filters()` exists but no plot |
| 5.2 | Visualize ScatNet wavelet filters (Morlet wavelets) | ❌ | Need to extract from Kymatio internals |
| 5.3 | Side-by-side comparison CNN vs ScatNet filters | ❌ | |
| 5.4 | If filters look noisy → add data augmentation and retrain | ❌ | |

---

## PART 6 — XAI METHODS ❌ ALL MISSING

All XAI files are empty (1-line placeholder). Must implement all 6:

| # | Method | File | Status | Library |
|---|--------|------|--------|---------|
| 6.1 | Vanilla Gradient / Saliency | `src/xai/saliency.py` | ❌ | Captum `Saliency` + manual |
| 6.2 | Integrated Gradients | `src/xai/integrated_gradients.py` | ❌ | Captum `IntegratedGradients` |
| 6.3 | GradCAM | `src/xai/gradcam.py` | ❌ | Captum `LayerGradCam` |
| 6.4 | DeepLIFT | `src/xai/deeplift.py` | ❌ | Captum `DeepLift` |
| 6.5 | Gradient × Input | `src/xai/gradient_based.py` | ❌ | Captum `InputXGradient` |
| 6.6 | **Custom from scratch** (Guided Backprop) | `src/xai/custom_method.py` | ❌ | Pure PyTorch hooks |

**IMPORTANT exam requirements for XAI:**
- Apply all 6 to **both CNN and ScatNet**
- Custom method (6.6) must be compared with its Captum equivalent (`GuidedBackprop`)
- Attributions must be **overlapped on the original image** (not shown separately)
- Show at least **2 images per class** (2 fractured + 2 not-fractured) for both models
- Discuss: which method works best? Which can't work on ScatNet? (GradCAM needs conv layers)

---

## PART 7 — NOTEBOOK (MAIN PIPELINE)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 7.1 | Single runnable Colab notebook `bone_fracture_detection_cnn.ipynb` | ⚠️ | Exists but likely incomplete |
| 7.2 | Mount Google Drive + install deps cell | ❌ | Add at top of notebook |
| 7.3 | Section 1: Data loading + visualization | ❌ |  |
| 7.4 | Section 2: CNN training with k-fold CV | ❌ | |
| 7.5 | Section 3: ScatNet training with k-fold CV | ❌ | |
| 7.6 | Section 4: Filter visualization (CNN vs ScatNet) | ❌ | |
| 7.7 | Section 5: Test set evaluation | ❌ | |
| 7.8 | Section 6: XAI methods on CNN | ❌ | |
| 7.9 | Section 7: XAI methods on ScatNet | ❌ | |
| 7.10 | Section 8: Custom XAI vs Captum comparison | ❌ | |
| 7.11 | Section 9: Results discussion | ❌ | |
| 7.12 | GPU runtime check (`torch.cuda.is_available()`) | ❌ | |

---

## PART 8 — PERFORMANCE TARGETS

| Requirement | Target | Status |
|-------------|--------|--------|
| Test accuracy | ≥ 75% | ❌ Not yet evaluated |
| F1 score reported | required | ❌ |
| Learning curves (train+val in same plot) | required | ❌ |
| Confusion matrix on test set | required | ❌ |

---

## SUMMARY OF CRITICAL GAPS (priority order)

1. ❌ **ScatNet** — entire model missing, blocks Parts 3,5,6,7
2. ❌ **All 6 XAI methods** — all files empty
3. ❌ **F1 score in CV** — cross_validation.py missing F1 computation
4. ❌ **Filter visualization** — no plotting code
5. ❌ **Data augmentation** — needed for good filters
6. ⚠️ **CV compatibility for ScatNet** — train_kfold_cv hardcodes CNN-specific params
7. ❌ **Main notebook** — needs full restructuring as single Colab pipeline

---

## FILES TO CREATE / MODIFY

```
MODIFY:
  src/models/scatnet.py          ← full implementation
  src/training/cross_validation.py ← add F1, fix ScatNet compat
  src/xai/saliency.py            ← implement
  src/xai/integrated_gradients.py ← implement
  src/xai/gradcam.py             ← implement
  src/xai/deeplift.py            ← implement
  src/xai/gradient_based.py      ← implement
  src/xai/custom_method.py       ← implement (Guided Backprop from scratch)

CREATE:
  src/visualization/filters.py   ← CNN + ScatNet filter plotting
  notebooks/bone_fracture_full_pipeline.ipynb ← complete Colab notebook
```
