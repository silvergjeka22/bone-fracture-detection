import torch
import torch.nn as nn
import numpy as np
import time
import copy
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from torch.utils.data import Subset, DataLoader
from .trainer import train_epoch, validate_epoch


def train_kfold_cv(model_class, train_dataset, model_kwargs=None, k_folds=5,
                   num_epochs=30, batch_size=32, learning_rate=0.001, device='cpu',
                   dropout_rate=0.5, model_name=None):
    """
    K-Fold Cross-Validation.

    Args:
        model_class: class (e.g. BoneFractureCNN or BoneFractureScatNet)
        train_dataset: PyTorch Dataset
        model_kwargs: dict passed to model_class(**model_kwargs).
                      If None, defaults to {'num_classes': 2, 'dropout_rate': dropout_rate}.
        k_folds: number of folds
        num_epochs: epochs per fold
        batch_size: training batch size
        learning_rate: initial learning rate (Adam)
        device: 'cpu', 'cuda', or 'mps'
        dropout_rate: legacy fallback when model_kwargs is None

    Returns:
        fold_results, fold_histories, mean_acc, std_acc, mean_f1, std_f1
    """
    if model_kwargs is None:
        model_kwargs = {'num_classes': 2, 'dropout_rate': dropout_rate}

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []
    fold_histories = []
    indices = list(range(len(train_dataset)))

    for fold, (train_ids, val_ids) in enumerate(kfold.split(indices)):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold + 1}/{k_folds}  —  train={len(train_ids)}, val={len(val_ids)}")
        print(f"{'='*60}")

        train_loader = DataLoader(
            Subset(train_dataset, train_ids),
            batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
        )
        val_loader = DataLoader(
            Subset(train_dataset, val_ids),
            batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
        )

        model = model_class(**model_kwargs).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0.0
        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(model.state_dict())
        best_preds, best_labels = [], []

        for epoch in range(num_epochs):
            t0 = time.time()
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                best_preds = val_preds
                best_labels = val_labels

            print(f"  [{epoch+1:2d}/{num_epochs}]  "
                  f"train loss={train_loss:.4f} acc={train_acc:.4f}  |  "
                  f"val   loss={val_loss:.4f} acc={val_acc:.4f}  "
                  f"({time.time()-t0:.1f}s)")

        model.load_state_dict(best_model_wts)
        fold_f1 = f1_score(best_labels, best_preds, average='binary', zero_division=0)

        fold_results.append({
            'fold': fold + 1,
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'best_val_f1': fold_f1,
            'final_train_acc': history['train_acc'][-1],
            'model': model,
        })
        fold_histories.append(history)

        print(f"\n  Fold {fold+1} best → acc={best_val_acc:.4f}  f1={fold_f1:.4f}")

    mean_acc = np.mean([r['best_val_acc'] for r in fold_results])
    std_acc  = np.std( [r['best_val_acc'] for r in fold_results])
    mean_f1  = np.mean([r['best_val_f1']  for r in fold_results])
    std_f1   = np.std( [r['best_val_f1']  for r in fold_results])

    print(f"\n{'='*60}")
    print("  K-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    for r in fold_results:
        print(f"  Fold {r['fold']}: acc={r['best_val_acc']:.4f}  f1={r['best_val_f1']:.4f}")
    print(f"\n  Mean Accuracy : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Mean F1 Score : {mean_f1:.4f} ± {std_f1:.4f}")

    if model_name:
        torch.save(model.state_dict(), f'{model_name}_best_fold_model.pth')

    return fold_results, fold_histories, mean_acc, std_acc, mean_f1, std_f1
