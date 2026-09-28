# Agent Prompt — Fix Full Pipeline (Bone Fracture Detection)

Pass this entire prompt to your coding agent (Claude Code, Cursor, etc.).

---

## CONTEXT

This is a University of Verona MSc AI exam project. The task: binary image classification (bone X-ray fractured vs not-fractured) using PyTorch, with ScatNet comparison and 6 XAI methods. The codebase structure exists but most implementations are missing or broken. You must fix everything and produce a single runnable Google Colab notebook.

**Libraries:** PyTorch, Captum, Kymatio, scikit-learn, matplotlib

**Codebase root:** `bone-fracture-detect   ion/`

**Dataset:** Kaggle "Bone Fracture Multi-Region X-ray Data" — folder structure:
```
data/
  train/
    fractured/    ← X-ray images
    not fractured/
  test/
    fractured/
    not fractured/
```
Use `gdown` or manual download. In the notebook, assume data is in `/content/drive/MyDrive/bone_fracture_data/` on Colab.

---

## WHAT TO BUILD

### STEP 1 — Fix `src/models/scatnet.py`

Implement `BoneFractureScatNet` using Kymatio. Requirements:
- Use `Scattering2D(J=2, shape=(224, 224))` from `kymatio.torch`
- Input images are RGB (3 channels) — convert to grayscale inside forward (mean of channels)
- Scattering output shape for J=2, shape=(224,224): `(B, 81, 56, 56)` — flatten to `(B, 81*56*56)` then use AdaptiveAvgPool to reduce
- **CRITICAL:** The FC classifier must be identical to CNN: `fc1(in→512) → ReLU → Dropout → fc2(512→128) → ReLU → Dropout → fc3(128→2)`. Only `in_features` of fc1 differs.
- Add `get_scatnet_filters(self)` method that returns the Morlet wavelet filters from `self.scattering.filters` for visualization
- Use this exact signature: `__init__(self, num_classes=2, J=2, L=8, dropout_rate=0.5)`

```python
from kymatio.torch import Scattering2D
import torch, torch.nn as nn

class BoneFractureScatNet(nn.Module):
    def __init__(self, num_classes=2, J=2, L=8, dropout_rate=0.5):
        super().__init__()
        self.J = J
        self.L = L
        self.scattering = Scattering2D(J=J, shape=(224, 224))
        # Compute flattened size: use AdaptiveAvgPool2d to get fixed (4,4) spatial
        # scattering output channels = 1 + J*L + J*(J-1)/2*L^2 (approx 81 for J=2,L=8)
        scat_channels = 1 + J * L + (J * (J - 1) // 2) * L * L  # = 81
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        flatten_size = scat_channels * 4 * 4  # 81 * 16 = 1296
        # Same classifier as CNN
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(flatten_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, 3, 224, 224) → grayscale → (B, 224, 224)
        x = x.mean(dim=1)
        # Scattering: (B, 224, 224) → (B, scat_channels, H', W')
        x = self.scattering(x)
        # Pool to fixed size
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # Same FC head as CNN
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def get_scatnet_filters(self):
        # Returns list of wavelet filter banks for visualization
        filters = self.scattering.filters
        return filters
```

---

### STEP 2 — Fix `src/training/cross_validation.py`

Add F1 score computation. Changes needed:
1. Import `f1_score` from sklearn
2. After each fold's best model is selected, run one final `validate_epoch` pass to get `all_preds, all_labels`
3. Compute `f1 = f1_score(all_labels, all_preds, average='binary')`
4. Store in `fold_results` and report mean F1 at end
5. Make `train_kfold_cv` accept `model_kwargs={}` instead of hardcoded `dropout_rate`, so it works for both CNN and ScatNet:

```python
model = model_class(**model_kwargs)
```

Call signature becomes:
```python
# For CNN:
train_kfold_cv(BoneFractureCNN, train_dataset, model_kwargs={'num_classes':2,'dropout_rate':0.5}, ...)
# For ScatNet:
train_kfold_cv(BoneFractureScatNet, train_dataset, model_kwargs={'num_classes':2,'J':2,'L':8,'dropout_rate':0.5}, ...)
```

Return signature: `fold_results, fold_histories, mean_acc, std_acc, mean_f1, std_f1`

---

### STEP 3 — Implement all XAI methods

#### `src/xai/saliency.py` — Vanilla Saliency
```python
# Manual: compute gradient of output w.r.t. input, take abs
# Also wrap Captum Saliency for comparison
from captum.attr import Saliency

def compute_saliency_manual(model, image_tensor, target_class):
    """Pure PyTorch saliency map."""
    image_tensor = image_tensor.unsqueeze(0).requires_grad_(True)
    model.eval()
    output = model(image_tensor)
    model.zero_grad()
    output[0, target_class].backward()
    saliency = image_tensor.grad.data.abs().squeeze()
    return saliency.mean(dim=0).cpu().numpy()  # (H, W)

def compute_saliency_captum(model, image_tensor, target_class):
    saliency = Saliency(model)
    attribution = saliency.attribute(image_tensor.unsqueeze(0), target=target_class)
    return attribution.squeeze().mean(dim=0).detach().cpu().numpy()
```

#### `src/xai/integrated_gradients.py` — Integrated Gradients
```python
from captum.attr import IntegratedGradients

def compute_integrated_gradients(model, image_tensor, target_class, n_steps=50):
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(image_tensor.unsqueeze(0))
    attribution = ig.attribute(image_tensor.unsqueeze(0), baseline, 
                               target=target_class, n_steps=n_steps)
    attr = attribution.squeeze().mean(dim=0).detach().cpu().numpy()
    return attr
```

#### `src/xai/gradcam.py` — GradCAM
```python
# Use Captum LayerGradCam on CNN's conv4 layer
# NOTE: GradCAM cannot be applied to ScatNet (no conv layers) — document this
from captum.attr import LayerGradCam
import torch.nn.functional as F

def compute_gradcam(model, image_tensor, target_class, target_layer):
    """target_layer: e.g., model.conv4 for CNN. Pass None for ScatNet."""
    if target_layer is None:
        return None  # GradCAM not applicable to ScatNet
    lgc = LayerGradCam(model, target_layer)
    attribution = lgc.attribute(image_tensor.unsqueeze(0), target=target_class)
    # Upsample to input size
    attr = F.interpolate(attribution, size=(224, 224), mode='bilinear', align_corners=False)
    return attr.squeeze().detach().cpu().numpy()
```

#### `src/xai/deeplift.py` — DeepLIFT
```python
from captum.attr import DeepLift

def compute_deeplift(model, image_tensor, target_class):
    dl = DeepLift(model)
    baseline = torch.zeros_like(image_tensor.unsqueeze(0))
    attribution = dl.attribute(image_tensor.unsqueeze(0), baseline, target=target_class)
    return attribution.squeeze().mean(dim=0).detach().cpu().numpy()
```

#### `src/xai/gradient_based.py` — Gradient × Input
```python
from captum.attr import InputXGradient

def compute_gradient_x_input(model, image_tensor, target_class):
    gxi = InputXGradient(model)
    attribution = gxi.attribute(image_tensor.unsqueeze(0), target=target_class)
    return attribution.squeeze().mean(dim=0).detach().cpu().numpy()
```

#### `src/xai/custom_method.py` — Guided Backpropagation FROM SCRATCH ⭐

This is the "custom from scratch" method. Must be compared against Captum's `GuidedBackprop`.

```python
import torch
import torch.nn as nn
import numpy as np
from captum.attr import GuidedBackprop  # for comparison only

class GuidedBackpropScratch:
    """
    Guided Backpropagation implemented from scratch using PyTorch hooks.
    
    Key idea: during backprop, only pass gradients that are:
    1. Positive in the gradient (standard ReLU backprop rule)
    2. Positive in the forward activation (guided part)
    
    Reference: Springenberg et al. (2014) "Striving for Simplicity"
    """
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register backward hooks on all ReLU layers to implement guided backprop."""
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                hook = module.register_backward_hook(self._guided_relu_backward)
                self.hooks.append(hook)
    
    @staticmethod
    def _guided_relu_backward(module, grad_input, grad_output):
        """
        Modify gradients: only keep positive gradients where forward activation was positive.
        grad_output[0] contains the gradients flowing back.
        """
        return (torch.clamp(grad_output[0], min=0.0),)
    
    def attribute(self, image_tensor, target_class):
        """
        Compute Guided Backprop attribution.
        
        Args:
            image_tensor: (1, C, H, W) tensor with requires_grad=True
            target_class: int
        Returns:
            numpy array (H, W) attribution map
        """
        self.model.eval()
        image_tensor = image_tensor.clone().requires_grad_(True)
        
        # Forward pass
        output = self.model(image_tensor)
        
        # Backward pass on target class
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Get gradients at input
        attribution = image_tensor.grad.data
        attribution = attribution.squeeze().mean(dim=0)
        attribution = torch.clamp(attribution, min=0)  # only positive attributions
        
        return attribution.cpu().numpy()
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def compute_guided_backprop_scratch(model, image_tensor, target_class):
    """Compute Guided Backprop from scratch."""
    gbp = GuidedBackpropScratch(model)
    attr = gbp.attribute(image_tensor.unsqueeze(0), target_class)
    gbp.remove_hooks()
    return attr


def compute_guided_backprop_captum(model, image_tensor, target_class):
    """Compute Guided Backprop using Captum (for comparison)."""
    gbp = GuidedBackprop(model)
    attribution = gbp.attribute(image_tensor.unsqueeze(0), target=target_class)
    attr = attribution.squeeze().mean(dim=0).detach().cpu().numpy()
    return np.maximum(attr, 0)


def compare_guided_backprop(model, image_tensor, target_class):
    """
    Compare custom vs Captum Guided Backprop.
    Returns dict with both attributions and correlation coefficient.
    """
    custom = compute_guided_backprop_scratch(model, image_tensor, target_class)
    captum = compute_guided_backprop_captum(model, image_tensor, target_class)
    
    # Pearson correlation as similarity metric
    corr = np.corrcoef(custom.flatten(), captum.flatten())[0, 1]
    
    return {
        'custom': custom,
        'captum': captum,
        'correlation': corr
    }
```

---

### STEP 4 — Create `src/visualization/filters.py`

```python
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_cnn_filters(model, save_path=None):
    """Visualize CNN conv1 filters (32 filters, show as RGB patches)."""
    filters = model.get_conv1_filters()  # (32, 3, 3, 3)
    filters = filters.cpu().numpy()
    
    # Normalize each filter to [0,1]
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min + 1e-8)
    
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle('CNN Conv1 Filters (32 filters × 3 channels)', fontsize=14)
    
    for i, ax in enumerate(axes.flat):
        if i < filters.shape[0]:
            # Show as RGB: (3, 3, 3) → (3, 3, 3) transposed to (3, 3, 3)
            f = np.transpose(filters[i], (1, 2, 0))
            ax.imshow(f)
            ax.set_title(f'F{i+1}', fontsize=7)
        ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_scatnet_filters(model, save_path=None):
    """Visualize ScatNet Morlet wavelet filters."""
    try:
        filters = model.get_scatnet_filters()
        # filters is a list of filter banks from Kymatio
        # First bank: psi filters (oriented wavelets)
        # Extract real parts for visualization
        
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        fig.suptitle('ScatNet Morlet Wavelet Filters', fontsize=14)
        
        if hasattr(filters, '__len__') and len(filters) > 0:
            psi_filters = filters[0] if isinstance(filters, list) else filters
            count = 0
            for ax in axes.flat:
                if count < len(psi_filters):
                    f = psi_filters[count]
                    if isinstance(f, dict):
                        f = f.get('levels', [None])[0]
                    if f is not None:
                        f_np = f.cpu().numpy() if hasattr(f, 'cpu') else np.array(f)
                        f_np = np.real(f_np) if np.iscomplexobj(f_np) else f_np
                        if f_np.ndim > 2:
                            f_np = f_np[0]
                        ax.imshow(f_np, cmap='RdBu_r')
                    count += 1
                ax.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
    except Exception as e:
        print(f"Could not visualize ScatNet filters: {e}")
        print("Plotting theoretical Morlet wavelet instead...")
        _plot_theoretical_morlet()


def _plot_theoretical_morlet():
    """Fallback: plot theoretical 2D Morlet wavelets at different orientations."""
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    fig, axes = plt.subplots(1, 8, figsize=(16, 2))
    fig.suptitle('Morlet Wavelets (ScatNet, 8 orientations)', fontsize=12)
    
    x = np.linspace(-3, 3, 64)
    y = np.linspace(-3, 3, 64)
    X, Y = np.meshgrid(x, y)
    sigma = 1.0
    k0 = 5.0
    
    for i, (ax, angle) in enumerate(zip(axes, angles)):
        Xr = X * np.cos(angle) + Y * np.sin(angle)
        gaussian = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        morlet = gaussian * np.cos(k0 * Xr)
        ax.imshow(morlet, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f'{np.degrees(angle):.0f}°', fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_attribution_overlay(image_tensor, attribution_map, title='Attribution', 
                              alpha=0.5, cmap='hot', save_path=None):
    """
    Overlay attribution map on original image.
    
    Args:
        image_tensor: (C, H, W) tensor (normalized)
        attribution_map: (H, W) numpy array
        title: plot title
        alpha: overlay transparency
    """
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    # Normalize attribution to [0,1]
    attr = attribution_map.copy()
    attr_min, attr_max = attr.min(), attr.max()
    if attr_max > attr_min:
        attr = (attr - attr_min) / (attr_max - attr_min)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(attr, cmap=cmap)
    axes[1].set_title('Attribution Map')
    axes[1].axis('off')
    
    axes[2].imshow(img)
    axes[2].imshow(attr, cmap=cmap, alpha=alpha)
    axes[2].set_title(f'{title} (Overlay)')
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_all_xai_comparison(image_tensor, attributions_dict, model_name='CNN', 
                             class_name='Fractured', save_path=None):
    """
    Plot all 6 XAI methods overlapped on one image in a grid.
    
    Args:
        attributions_dict: {'Saliency': np.array, 'IntGrad': np.array, ...}
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    methods = list(attributions_dict.keys())
    n = len(methods)
    
    fig, axes = plt.subplots(2, (n + 2) // 2, figsize=(4 * ((n + 2) // 2), 8))
    fig.suptitle(f'XAI Methods Comparison — {model_name} — {class_name}', fontsize=14)
    axes = axes.flat
    
    # First cell: original
    next(axes).imshow(img)
    
    for ax, method in zip(axes, methods):
        attr = attributions_dict[method]
        if attr is None:
            ax.text(0.5, 0.5, f'{method}\nN/A', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue
        attr_n = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        ax.imshow(img)
        ax.imshow(attr_n, cmap='hot', alpha=0.5)
        ax.set_title(method, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
```

---

### STEP 5 — Create the main Colab notebook

Create `notebooks/bone_fracture_full_pipeline.ipynb` as a complete, runnable Google Colab notebook with these sections. Write all cells in order:

**Cell 1 — Setup:**
```python
# Install dependencies
!pip install kymatio captum -q

import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Cell 2 — Mount Drive + clone repo:**
```python
from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.insert(0, '/content/drive/MyDrive/bone-fracture-detection')
# Or: !git clone <repo_url> && sys.path.insert(0, '/content/bone-fracture-detection')
```

**Cell 3 — Imports:**
```python
import torch, torch.nn as nn
import numpy as np, matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from src.data.datasets import BoneFractureDataset
from src.models.cnn import BoneFractureCNN
from src.models.scatnet import BoneFractureScatNet
from src.training.cross_validation import train_kfold_cv
from src.training.evaluator import evaluate_model
from src.visualization.filters import plot_cnn_filters, plot_scatnet_filters
from src.visualization.filters import plot_all_xai_comparison, plot_attribution_overlay
from src.xai.saliency import compute_saliency_captum
from src.xai.integrated_gradients import compute_integrated_gradients
from src.xai.gradcam import compute_gradcam
from src.xai.deeplift import compute_deeplift
from src.xai.gradient_based import compute_gradient_x_input
from src.xai.custom_method import compare_guided_backprop, compute_guided_backprop_scratch
```

**Cell 4 — Data loading with augmentation:**
```python
DATA_ROOT = '/content/drive/MyDrive/bone_fracture_data'

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = BoneFractureDataset(f'{DATA_ROOT}/train', transform=train_transform)
test_dataset  = BoneFractureDataset(f'{DATA_ROOT}/test',  transform=test_transform)
test_loader   = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Show sample images
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for class_name, row in zip(train_dataset.class_names, axes):
    samples = train_dataset.get_sample_by_class(class_name, num_samples=4)
    for ax, (img_path, _) in zip(row, samples):
        from PIL import Image
        ax.imshow(Image.open(img_path).convert('RGB'))
        ax.set_title(class_name)
        ax.axis('off')
plt.suptitle('Sample Training Images')
plt.show()
```

**Cell 5 — Train CNN with k-fold CV:**
```python
print("=== Training CNN with 5-Fold Cross-Validation ===")
cnn_fold_results, cnn_histories, cnn_mean_acc, cnn_std_acc, cnn_mean_f1, cnn_std_f1 = train_kfold_cv(
    BoneFractureCNN, train_dataset,
    model_kwargs={'num_classes': 2, 'dropout_rate': 0.5},
    k_folds=5, num_epochs=30, batch_size=32, learning_rate=1e-3, device=device
)
print(f"\nCNN CV Results: Acc = {cnn_mean_acc:.4f} ± {cnn_std_acc:.4f} | F1 = {cnn_mean_f1:.4f} ± {cnn_std_f1:.4f}")
```

**Cell 6 — Plot CNN learning curves:**
```python
# Plot all folds in same figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('CNN — Learning Curves (5-Fold CV)')
colors = plt.cm.tab10.colors

for i, hist in enumerate(cnn_histories):
    epochs = range(1, len(hist['train_loss']) + 1)
    ax1.plot(epochs, hist['train_loss'], '--', color=colors[i], alpha=0.7, label=f'Fold {i+1} Train')
    ax1.plot(epochs, hist['val_loss'],   '-',  color=colors[i], alpha=0.7, label=f'Fold {i+1} Val')
    ax2.plot(epochs, hist['train_acc'],  '--', color=colors[i], alpha=0.7)
    ax2.plot(epochs, hist['val_acc'],    '-',  color=colors[i], alpha=0.7)

ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Loss'); ax1.legend(fontsize=6)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.set_title('Accuracy')
plt.tight_layout(); plt.show()
```

**Cell 7 — Train ScatNet with k-fold CV:**
```python
print("=== Training ScatNet with 5-Fold Cross-Validation ===")
scat_fold_results, scat_histories, scat_mean_acc, scat_std_acc, scat_mean_f1, scat_std_f1 = train_kfold_cv(
    BoneFractureScatNet, train_dataset,
    model_kwargs={'num_classes': 2, 'J': 2, 'L': 8, 'dropout_rate': 0.5},
    k_folds=5, num_epochs=30, batch_size=32, learning_rate=1e-3, device=device
)
print(f"\nScatNet CV Results: Acc = {scat_mean_acc:.4f} ± {scat_std_acc:.4f} | F1 = {scat_mean_f1:.4f} ± {scat_std_f1:.4f}")
# Plot same as CNN...
```

**Cell 8 — Select best models from CV:**
```python
# Best CNN = fold with highest val_acc
best_cnn_fold = max(cnn_fold_results, key=lambda x: x['best_val_acc'])
best_cnn = best_cnn_fold['model'].to(device)
best_cnn.eval()

best_scat_fold = max(scat_fold_results, key=lambda x: x['best_val_acc'])
best_scatnet = best_scat_fold['model'].to(device)
best_scatnet.eval()

print(f"Best CNN fold: {best_cnn_fold['fold']} (val_acc={best_cnn_fold['best_val_acc']:.4f})")
print(f"Best ScatNet fold: {best_scat_fold['fold']} (val_acc={best_scat_fold['best_val_acc']:.4f})")
```

**Cell 9 — Filter visualization:**
```python
# CNN filters
plot_cnn_filters(best_cnn, save_path='cnn_filters.png')

# ScatNet wavelet filters
plot_scatnet_filters(best_scatnet, save_path='scatnet_filters.png')
```

**Cell 10 — Test set evaluation:**
```python
from sklearn.metrics import classification_report

def evaluate_on_test(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='binary')
    return acc, f1, all_preds, all_labels

cnn_acc, cnn_f1, cnn_preds, cnn_true = evaluate_on_test(best_cnn, test_loader, device)
scat_acc, scat_f1, scat_preds, scat_true = evaluate_on_test(best_scatnet, test_loader, device)

print(f"CNN  — Test Acc: {cnn_acc:.4f} | F1: {cnn_f1:.4f}")
print(f"ScatNet — Test Acc: {scat_acc:.4f} | F1: {scat_f1:.4f}")

# Confusion matrices side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay(confusion_matrix(cnn_true, cnn_preds), 
                       display_labels=['Not Fractured','Fractured']).plot(ax=ax1)
ax1.set_title(f'CNN (Acc={cnn_acc:.3f}, F1={cnn_f1:.3f})')
ConfusionMatrixDisplay(confusion_matrix(scat_true, scat_preds),
                       display_labels=['Not Fractured','Fractured']).plot(ax=ax2)
ax2.set_title(f'ScatNet (Acc={scat_acc:.3f}, F1={scat_f1:.3f})')
plt.tight_layout(); plt.show()
```

**Cell 11 — Prepare XAI sample images (2 per class, both models):**
```python
# Get 2 fractured + 2 not-fractured samples
xai_samples = {}
for class_name in test_dataset.class_names:
    samples = test_dataset.get_sample_by_class(class_name, num_samples=2)
    xai_samples[class_name] = []
    for img_path, label in samples:
        from PIL import Image
        img = test_transform(Image.open(img_path).convert('RGB'))
        xai_samples[class_name].append((img, label, img_path))
```

**Cell 12 — Run all 6 XAI methods on CNN:**
```python
xai_methods_cnn = {}

for class_name, samples in xai_samples.items():
    for img_tensor, label, img_path in samples:
        t = img_tensor.unsqueeze(0).to(device)
        target = label
        
        attributions = {
            'Saliency':       compute_saliency_captum(best_cnn, img_tensor.to(device), target),
            'IntGrad':        compute_integrated_gradients(best_cnn, img_tensor.to(device), target),
            'GradCAM':        compute_gradcam(best_cnn, img_tensor.to(device), target, best_cnn.conv4),
            'DeepLIFT':       compute_deeplift(best_cnn, img_tensor.to(device), target),
            'Grad×Input':     compute_gradient_x_input(best_cnn, img_tensor.to(device), target),
            'Guided BP':      compute_guided_backprop_scratch(best_cnn, img_tensor.to(device), target),
        }
        plot_all_xai_comparison(img_tensor, attributions, 
                                 model_name='CNN', class_name=class_name,
                                 save_path=f'xai_cnn_{class_name}_{img_path[-10:]}.png')
```

**Cell 13 — Run all 6 XAI methods on ScatNet:**
```python
# Same as Cell 12 but with best_scatnet
# NOTE: GradCAM → pass None for target_layer (ScatNet has no conv layers → N/A)
for class_name, samples in xai_samples.items():
    for img_tensor, label, img_path in samples:
        attributions = {
            'Saliency':   compute_saliency_captum(best_scatnet, img_tensor.to(device), label),
            'IntGrad':    compute_integrated_gradients(best_scatnet, img_tensor.to(device), label),
            'GradCAM':    None,  # NOT APPLICABLE — ScatNet has no convolutional layers
            'DeepLIFT':   compute_deeplift(best_scatnet, img_tensor.to(device), label),
            'Grad×Input': compute_gradient_x_input(best_scatnet, img_tensor.to(device), label),
            'Guided BP':  compute_guided_backprop_scratch(best_scatnet, img_tensor.to(device), label),
        }
        plot_all_xai_comparison(img_tensor, attributions,
                                 model_name='ScatNet', class_name=class_name)
```

**Cell 14 — Compare custom Guided BP vs Captum:**
```python
# Pick one sample image
sample_img, sample_label, _ = xai_samples[test_dataset.class_names[0]][0]
sample_img = sample_img.to(device)

comparison = compare_guided_backprop(best_cnn, sample_img, sample_label)
print(f"Correlation between custom and Captum Guided Backprop: {comparison['correlation']:.4f}")

# Plot side-by-side
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle('Guided Backpropagation: Custom (scratch) vs Captum')

mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])
img_np = sample_img.cpu().numpy().transpose(1,2,0) * std + mean
img_np = np.clip(img_np, 0, 1)

axes[0].imshow(img_np); axes[0].set_title('Original'); axes[0].axis('off')
axes[1].imshow(comparison['custom'], cmap='hot'); axes[1].set_title(f'Custom (scratch)'); axes[1].axis('off')
axes[2].imshow(comparison['captum'], cmap='hot'); axes[2].set_title(f'Captum (corr={comparison["correlation"]:.3f})'); axes[2].axis('off')
plt.tight_layout(); plt.show()
```

**Cell 15 — Results Discussion (Markdown cell):**
```
## Discussion

### Model Performance
- CNN vs ScatNet accuracy and F1 comparison
- CNN typically performs better on complex fracture patterns due to learned features
- ScatNet provides translation-invariant features with no learning required in feature extraction

### Filter Analysis
- CNN conv1 filters: expected edge detectors at various orientations
- ScatNet: Morlet wavelets at J=2 scales and L=8 orientations — mathematically defined
- If CNN filters appear noisy → indicates insufficient training data or need more augmentation

### XAI Methods Analysis
- **Saliency**: Fast but noisy — highlights pixel-level sensitivity
- **Integrated Gradients**: More robust — considers full gradient path from baseline
- **GradCAM**: Highlights class-discriminative regions (CNN only — NOT applicable to ScatNet which has no convolutional feature maps)
- **DeepLIFT**: Compares activations to reference — good for sparse attributions
- **Gradient × Input**: Combines gradient magnitude with input importance
- **Guided Backprop (custom)**: Sharp, high-resolution attributions focusing on positive contributions

### Custom vs Captum Comparison
- Pearson correlation between implementations should be > 0.95 if hooks are correctly applied
- Differences arise from edge cases in ReLU handling

### When GradCAM Cannot Be Applied
- GradCAM requires spatial feature maps from a convolutional layer
- ScatNet uses fixed wavelet transforms — no intermediate conv layers → GradCAM returns None
```

---

## CRITICAL IMPLEMENTATION NOTES

1. **All XAI functions must call `model.eval()` first**
2. **All `image_tensor` inputs to XAI must NOT have batch dimension** — add `.unsqueeze(0)` inside the function
3. **Captum requires `requires_grad=True`** on input tensors — handle inside each function
4. **ScatNet `scattering` must be moved to device**: call `self.scattering = self.scattering.to(device)` in `__init__` or in the notebook after instantiation
5. **GradCAM layer for CNN**: use `best_cnn.conv4` (last conv before flatten)
6. **F1 score**: use `average='binary'` since this is binary classification
7. **Attribution overlay**: always use `alpha=0.5` and `cmap='hot'` for consistency
8. **Learning curves**: train and val in the SAME subplot (not separate figures) — exam requirement

---

## FILE MODIFICATION CHECKLIST

```
✏️  src/models/scatnet.py           — full implementation (replace NotImplementedError)
✏️  src/training/cross_validation.py — add F1, change model_kwargs signature
✏️  src/xai/saliency.py             — implement manual + captum
✏️  src/xai/integrated_gradients.py — implement with Captum
✏️  src/xai/gradcam.py              — implement with Captum LayerGradCam
✏️  src/xai/deeplift.py             — implement with Captum DeepLift
✏️  src/xai/gradient_based.py       — implement with Captum InputXGradient
✏️  src/xai/custom_method.py        — Guided Backprop from scratch + compare()
➕  src/visualization/filters.py    — NEW: filter plotting + XAI overlay utils
➕  notebooks/bone_fracture_full_pipeline.ipynb — NEW: complete Colab notebook
```

After implementing all the above, verify:
- `python -c "from src.models.scatnet import BoneFractureScatNet; m = BoneFractureScatNet(); print('ScatNet OK')"` 
- `python -c "from src.xai.custom_method import compare_guided_backprop; print('XAI OK')"`
- Run first 3 cells of notebook and confirm no import errors
