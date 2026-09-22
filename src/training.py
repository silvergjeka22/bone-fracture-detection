"""Training, k-fold cross-validation, test evaluation and saving/loading of results.

Files written to RESULTS_DIR:
    models/<name>.pth   weights of the best CV fold (used for testing and XAI)
    cv/<name>.json      per-fold metrics and learning curves
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

from .data import make_loader
from .models import build_model

POSITIVE_CLASS = 0  # 'fractured': F1 / precision / recall are computed for the class we must detect


def compute_metrics(y_true, y_pred):
    kw = dict(pos_label=POSITIVE_CLASS, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, **kw),
        "precision": precision_score(y_true, y_pred, **kw),
        "recall": recall_score(y_true, y_pred, **kw),
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (outputs.argmax(1) == labels).sum().item()
        seen += len(labels)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    """Mean loss (0 if no criterion), predictions and true labels over a whole loader."""
    model.eval()
    total_loss, preds, labels_all = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if criterion is not None:
            total_loss += criterion(outputs, labels).item() * len(labels)
        preds.append(outputs.argmax(1).cpu())
        labels_all.append(labels.cpu())
    preds, labels_all = torch.cat(preds).numpy(), torch.cat(labels_all).numpy()
    return total_loss / len(labels_all), preds, labels_all


def train_kfold_cv(name, train_set, eval_set, device, k_folds=5, epochs=10, batch_size=32,
                   lr=1e-3, results_dir="results", seed=42):
    """Stratified k-fold cross-validation on the training set.

    `train_set` and `eval_set` contain the same images: the first with augmentation (training
    folds), the second without (validation folds). In every fold the weights of the epoch with
    the best validation accuracy are kept; the best fold is saved as results/models/<name>.pth.
    """
    labels = np.array(train_set.labels)
    splits = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed).split(labels, labels)
    folds, histories = [], []
    best_acc, best_state = -1.0, None

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        print(f"\n=== {name} | fold {fold}/{k_folds} | train={len(train_idx)} val={len(val_idx)} ===")
        model = build_model(name).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        train_loader = make_loader(Subset(train_set, train_idx), batch_size, shuffle=True)
        val_loader = make_loader(Subset(eval_set, val_idx), batch_size)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        fold_best = {"val_acc": -1.0}
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, preds, y = evaluate(model, val_loader, device, criterion)
            val_metrics = compute_metrics(y, preds)
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_metrics["accuracy"])
            print(f"  [{epoch:2d}/{epochs}] train loss={train_loss:.4f} acc={train_acc:.4f} | "
                  f"val loss={val_loss:.4f} acc={val_metrics['accuracy']:.4f} ({time.time() - start:.0f}s)")

            if val_metrics["accuracy"] > fold_best["val_acc"]:
                fold_best = {"fold": fold, "epoch": epoch, "val_acc": val_metrics["accuracy"],
                             "val_f1": val_metrics["f1"]}
                fold_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"  best epoch {fold_best['epoch']}: acc={fold_best['val_acc']:.4f} f1={fold_best['val_f1']:.4f}")
        folds.append(fold_best)
        histories.append(history)
        if fold_best["val_acc"] > best_acc:
            best_acc, best_state = fold_best["val_acc"], fold_state
        del model, optimizer  # free GPU memory before the next fold

    accs = [f["val_acc"] for f in folds]
    f1s = [f["val_f1"] for f in folds]
    results = {
        "model": name, "folds": folds, "histories": histories,
        "mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs)),
        "mean_f1": float(np.mean(f1s)), "std_f1": float(np.std(f1s)),
        "best_fold": int(np.argmax(accs)) + 1,
    }
    print(f"\n{name}: mean acc = {results['mean_acc']:.4f} ± {results['std_acc']:.4f} | "
          f"mean F1 = {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")

    save_model(best_state, name, results_dir)
    cv_path = Path(results_dir) / "cv" / f"{name}.json"
    cv_path.parent.mkdir(parents=True, exist_ok=True)
    cv_path.write_text(json.dumps(results, indent=1))
    return results


def load_cv_results(name, results_dir="results"):
    path = Path(results_dir) / "cv" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: set TRAIN = True to run cross-validation.")
    return json.loads(path.read_text())


def save_model(state_dict, name, results_dir="results"):
    path = Path(results_dir) / "models" / f"{name}.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)
    print(f"saved {path}")


def load_model(name, results_dir, device):
    """Model with the saved weights of results/models/<name>.pth, in eval mode."""
    path = Path(results_dir) / "models" / f"{name}.pth"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: set TRAIN = True or copy the trained weights there.")
    model = build_model(name, pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def test_model(model, loader, device):
    """Metrics, confusion matrix and predictions on the test set."""
    _, preds, labels = evaluate(model, loader, device)
    return {**compute_metrics(labels, preds),
            "confusion_matrix": confusion_matrix(labels, preds).tolist(),
            "y_true": labels.tolist(), "y_pred": preds.tolist()}


@torch.no_grad()
def predict_labels(model, images, device):
    model.eval()
    return model(images.to(device)).argmax(1).cpu().numpy()


def cv_table(cv):
    """Mean ± std of the validation accuracy and F1 across folds, one row per model."""
    return pd.DataFrame({
        name: {"CV accuracy": f"{r['mean_acc']:.4f} ± {r['std_acc']:.4f}",
               "CV F1": f"{r['mean_f1']:.4f} ± {r['std_f1']:.4f}",
               "best fold": r["best_fold"]}
        for name, r in cv.items()
    }).T


def results_table(cv, test):
    """Cross-validation and test metrics side by side, best CV accuracy first."""
    return pd.DataFrame({
        name: {"CV accuracy": cv[name]["mean_acc"], "CV F1": cv[name]["mean_f1"],
               "test accuracy": test[name]["accuracy"], "test F1": test[name]["f1"],
               "test precision": test[name]["precision"], "test recall": test[name]["recall"]}
        for name in test
    }).T.round(4).sort_values("CV accuracy", ascending=False)


def select_best(cv, metric="mean_acc"):
    """Name of the model with the highest cross-validation score ('mean_acc' or 'mean_f1').

    The choice uses only the cross-validation on the training set: choosing on the test set
    would turn the test accuracy of the chosen model into an optimistic estimate.
    """
    best = max(cv, key=lambda name: cv[name][metric])
    print(f"best model: {best} (CV {metric} = {cv[best][metric]:.4f})")
    return best


def error_table(test, class_names):
    """Where each model goes wrong on the test set.

    'missed' = fractures predicted as not fractured (the costly error), 'false alarms' = the
    opposite, 'only this model' = test images that every other model classifies correctly.
    """
    wrong = {name: {i for i, (t, p) in enumerate(zip(r["y_true"], r["y_pred"])) if t != p}
             for name, r in test.items()}
    rows = {}
    for name, r in test.items():
        y_true, y_pred = np.array(r["y_true"]), np.array(r["y_pred"])
        others = set().union(*(wrong[m] for m in wrong if m != name))
        rows[name] = {
            f"missed {class_names[POSITIVE_CLASS]}": int(((y_true == POSITIVE_CLASS) & (y_pred != POSITIVE_CLASS)).sum()),
            "false alarms": int(((y_true != POSITIVE_CLASS) & (y_pred == POSITIVE_CLASS)).sum()),
            "total errors": len(wrong[name]),
            "only this model": len(wrong[name] - others),
        }
    print(f"{len(set.intersection(*wrong.values()))} test images are misclassified by every model")
    return pd.DataFrame(rows).T


def prediction_table(labels, preds, class_names):
    """True class vs the class predicted by each model, one row per image."""
    table = pd.DataFrame({"true": [class_names[i] for i in labels]})
    for name, p in preds.items():
        table[name] = [class_names[i] if i == t else f"{class_names[i]} ✗" for i, t in zip(p, labels)]
    return table
