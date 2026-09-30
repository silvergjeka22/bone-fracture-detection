"""Training and evaluation: k-fold cross-validation, final training, test metrics, statistical tests.

Protocol
    1. k-fold cross-validation on the TRAINING split -> mean accuracy and mean F1 (generalisation estimate).
       Folds are stratified by class and grouped by near-duplicate, so copies of the same X-ray
       are never on both sides of a split. Fold metrics are taken at the last epoch: nothing is
       chosen on the validation fold, so the estimate is not optimistic.
    2. Final model: trained on the whole training split; the VAL split picks the best epoch.
    3. TEST split: used once, to evaluate the final models.

Files written to results_dir:  cv/<model>.json, final/<model>.json, models/<model>.pth
"""

import copy
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .data import make_loader
from .models import build_model
from .utils import count_parameters, set_seed

POSITIVE = 0  # 'fractured': the class we must detect. F1, precision and recall are computed for it.
METRICS = ("accuracy", "f1", "precision", "recall", "specificity", "auc")


def compute_metrics(y_true, probs):
    """Metrics from true labels (N,) and predicted class probabilities (N, 2)."""
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    y_pred = probs.argmax(1)
    kw = dict(pos_label=POSITIVE, zero_division=0)
    is_pos = y_true == POSITIVE
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, **kw),
        "precision": precision_score(y_true, y_pred, **kw),
        "recall": recall_score(y_true, y_pred, **kw),                   # sensitivity: fractures found
        "specificity": float(((y_pred != POSITIVE) & ~is_pos).sum() / max((~is_pos).sum(), 1)),
        "auc": roc_auc_score(is_pos, probs[:, POSITIVE]) if 0 < is_pos.sum() < len(is_pos) else float("nan"),
    }


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (outputs.argmax(1) == labels).sum().item()
        seen += len(labels)
    return total_loss / seen, correct / seen


@torch.no_grad()
def predict(model, loader, device):
    """Class probabilities (N, 2), true labels (N,) and mean cross-entropy over a loader."""
    model.eval()
    probs, labels_all, total_loss = [], [], 0.0
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        total_loss += nn.functional.cross_entropy(logits, labels.to(device), reduction="sum").item()
        probs.append(logits.softmax(1).cpu())
        labels_all.append(labels)
    labels_all = torch.cat(labels_all).numpy()
    return torch.cat(probs).numpy(), labels_all, total_loss / len(labels_all)


def fit(model, train_loader, val_loader, device, epochs=15, lr=1e-3, weight_decay=1e-4, keep_best=False, tag=""):
    """Train with AdamW and a cosine learning-rate schedule, evaluating on `val_loader` after every epoch.

    keep_best=True restores the weights of the epoch with the best validation accuracy.
    Returns the history (train and validation loss/accuracy per epoch) and the kept epoch.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "seconds")}
    best_acc, best_epoch, best_state = -1.0, epochs, None
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        probs, y, val_loss = predict(model, val_loader, device)
        val = compute_metrics(y, probs)
        scheduler.step()
        for key, value in zip(history, (train_loss, train_acc, val_loss, val["accuracy"], val["f1"], time.time() - start)):
            history[key].append(float(value))
        print(f"{tag}[{epoch:2d}/{epochs}] train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val['accuracy']:.4f} f1 {val['f1']:.4f} ({history['seconds'][-1]:.0f}s)")
        if keep_best and val["accuracy"] > best_acc:
            best_acc, best_epoch = val["accuracy"], epoch
            best_state = copy.deepcopy(model.state_dict())
    if keep_best:
        model.load_state_dict(best_state)
    return history, best_epoch


def cross_validate(name, train_set, groups, device, k_folds=5, epochs=15, batch_size=32, lr=1e-3,
                   weight_decay=1e-4, seed=42, num_workers=2, image_size=224, pretrained=True, results_dir=None):
    """Stratified, group-aware k-fold cross-validation of one model on the training split."""
    labels = train_set.labels
    splitter = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    folds, histories = [], []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups), start=1):
        print(f"=== {name} | fold {fold}/{k_folds} | train {len(train_idx)} | validation {len(val_idx)} ===")
        set_seed(seed + fold)
        model = build_model(name, image_size, pretrained).to(device)
        train_loader = make_loader(train_set, train_idx, augment=True, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers)
        val_loader = make_loader(train_set, val_idx, batch_size=2 * batch_size, num_workers=num_workers)
        history, _ = fit(model, train_loader, val_loader, device, epochs, lr, weight_decay, tag=f"fold {fold} ")
        probs, y, _ = predict(model, val_loader, device)
        folds.append({"fold": fold, "n_train": len(train_idx), "n_val": len(val_idx), **compute_metrics(y, probs)})
        histories.append(history)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    table = pd.DataFrame(folds)
    result = {"model": name, "folds": folds, "histories": histories,
              "mean": {m: float(table[m].mean()) for m in METRICS},
              "std": {m: float(table[m].std(ddof=0)) for m in METRICS},
              "seconds_per_epoch": float(np.mean([s for h in histories for s in h["seconds"]])),
              "settings": dict(k_folds=k_folds, epochs=epochs, batch_size=batch_size, lr=lr,
                               weight_decay=weight_decay, seed=seed)}
    print(f"{name}: CV accuracy {result['mean']['accuracy']:.4f} ± {result['std']['accuracy']:.4f} | "
          f"CV F1 {result['mean']['f1']:.4f} ± {result['std']['f1']:.4f}")
    if results_dir is not None:
        save_json(result, Path(results_dir) / "cv" / f"{name}.json")
    return result


def train_final(name, train_set, val_set, device, epochs=15, batch_size=32, lr=1e-3, weight_decay=1e-4,
                seed=42, num_workers=2, image_size=224, pretrained=True, results_dir=None):
    """Final model trained on the whole training split, best epoch chosen on the val split."""
    print(f"=== {name} | final training on {len(train_set)} images, validation {len(val_set)} ===")
    set_seed(seed)
    model = build_model(name, image_size, pretrained).to(device)
    train_loader = make_loader(train_set, augment=True, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_loader(val_set, batch_size=2 * batch_size, num_workers=num_workers)
    history, best_epoch = fit(model, train_loader, val_loader, device, epochs, lr, weight_decay, keep_best=True,
                              tag="final ")
    info = {"model": name, "history": history, "best_epoch": best_epoch,
            "seconds_per_epoch": float(np.mean(history["seconds"])),
            "parameters": count_parameters(model)}
    if results_dir is not None:
        save_model(model, name, results_dir)
        save_json(info, Path(results_dir) / "final" / f"{name}.json")
    return model.eval(), info


def evaluate_test(model, test_set, device, leaked=None, batch_size=64):
    """Test metrics, 95% bootstrap interval of the accuracy, confusion matrix and predictions.

    `leaked` marks test images with a near-duplicate in the training split; the metrics on the
    others ('clean' test images) show how much those copies inflate the test score.
    """
    loader = make_loader(test_set, batch_size=batch_size)
    start = time.time()
    probs, y, _ = predict(model, loader, device)
    ms_per_image = 1000 * (time.time() - start) / len(y)
    y_pred = probs.argmax(1)
    result = {**compute_metrics(y, probs),
              "accuracy_ci": bootstrap_ci(y, y_pred),
              "confusion_matrix": confusion_matrix(y, y_pred, labels=[0, 1]).tolist(),
              "y_true": y.tolist(), "y_pred": y_pred.tolist(), "prob_positive": probs[:, POSITIVE].tolist(),
              "ms_per_image": ms_per_image}
    if leaked is not None and (~leaked).any():
        result["clean"] = compute_metrics(y[~leaked], probs[~leaked])
        result["n_clean"] = int((~leaked).sum())
    return result


def bootstrap_ci(y_true, y_pred, n_boot=2000, level=0.95, seed=0):
    """Percentile bootstrap interval of the accuracy (resampling the test images)."""
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    rng = np.random.default_rng(seed)
    means = correct[rng.integers(0, len(correct), (n_boot, len(correct)))].mean(1)
    return [float(np.quantile(means, (1 - level) / 2)), float(np.quantile(means, (1 + level) / 2))]


def mcnemar(y_true, pred_a, pred_b):
    """Exact McNemar test: do models A and B have the same error rate on the same test images?"""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    only_a = int(((pred_a == y_true) & (pred_b != y_true)).sum())
    only_b = int(((pred_a != y_true) & (pred_b == y_true)).sum())
    p = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return only_a, only_b, float(p)


def mcnemar_table(test):
    """Pairwise McNemar tests: images only A / only B classifies correctly, p-value."""
    rows = {}
    for a, b in combinations(test, 2):
        only_a, only_b, p = mcnemar(test[a]["y_true"], test[a]["y_pred"], test[b]["y_pred"])
        rows[f"{a} vs {b}"] = {"only A right": only_a, "only B right": only_b, "p-value": round(p, 4),
                               "significant (p<0.05)": bool(p < 0.05)}
    return pd.DataFrame.from_dict(rows, orient="index")


# ----------------------------------------------------------------------------- tables and choice

def _pm(mean, std):
    return f"{100 * mean:.1f} ± {100 * std:.1f}"


def cv_table(cv):
    """Mean ± std over the folds (in %), one row per model."""
    return pd.DataFrame({name: {m: _pm(r["mean"][m], r["std"][m]) for m in METRICS} for name, r in cv.items()}).T


def test_table(test):
    rows = {}
    for name, r in test.items():
        lo, hi = r["accuracy_ci"]
        rows[name] = {"accuracy": f"{100 * r['accuracy']:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]",
                      **{m: round(100 * r[m], 1) for m in METRICS[1:]}}
        if "clean" in r:
            rows[name][f"accuracy on clean test ({r['n_clean']})"] = round(100 * r["clean"]["accuracy"], 1)
    return pd.DataFrame(rows).T


def comparison_table(cv, test, final):
    """Everything needed to choose a model, numeric, one row per model."""
    return pd.DataFrame({name: {
        "CV accuracy": cv[name]["mean"]["accuracy"], "CV F1": cv[name]["mean"]["f1"],
        "test accuracy": test[name]["accuracy"], "test F1": test[name]["f1"],
        "test recall": test[name]["recall"], "test AUC": test[name]["auc"],
        "parameters (M)": final[name]["parameters"] / 1e6,
        "s / epoch": final[name]["seconds_per_epoch"], "ms / image": test[name]["ms_per_image"],
    } for name in test}).T.astype(float).round(4)


def select_best(cv, metric="f1"):
    """Model with the highest mean cross-validation `metric`.

    Chosen on the training split only: choosing on the test set would make the test score of
    the chosen model optimistic.
    """
    best = max(cv, key=lambda name: cv[name]["mean"][metric])
    print(f"best model: {best} (CV {metric} = {cv[best]['mean'][metric]:.4f})")
    return best


def error_table(test, class_names):
    """Where each model goes wrong: missed fractures (the costly error), false alarms, errors no other model makes."""
    wrong = {n: {i for i, (t, p) in enumerate(zip(r["y_true"], r["y_pred"])) if t != p} for n, r in test.items()}
    rows = {}
    for name, r in test.items():
        y, p = np.array(r["y_true"]), np.array(r["y_pred"])
        others = set().union(*(wrong[m] for m in wrong if m != name))
        rows[name] = {f"missed {class_names[POSITIVE]}": int(((y == POSITIVE) & (p != POSITIVE)).sum()),
                      "false alarms": int(((y != POSITIVE) & (p == POSITIVE)).sum()),
                      "total errors": len(wrong[name]),
                      "errors no other model makes": len(wrong[name] - others)}
    table = pd.DataFrame(rows).T
    table.attrs["wrong for every model"] = len(set.intersection(*wrong.values()))
    return table


# ----------------------------------------------------------------------------- saving / loading

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the notebook with TRAIN = True first.")
    return json.loads(path.read_text())


def save_model(model, name, results_dir):
    path = Path(results_dir) / "models" / f"{name}.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(name, results_dir, device, image_size=224):
    """Model with the weights saved by train_final, in eval mode."""
    path = Path(results_dir) / "models" / f"{name}.pth"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the notebook with TRAIN = True first.")
    model = build_model(name, image_size, pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval()
