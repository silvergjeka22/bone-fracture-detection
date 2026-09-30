import json

import numpy as np
import pytest
import torch

from src import data, training
from src.models import build_model
from tests.conftest import SIZE


def test_metrics_fractured_is_positive():
    y = np.array([0, 0, 1, 1])
    probs = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.7, 0.3]])  # predictions 0, 1, 1, 0
    m = training.compute_metrics(y, probs)
    assert m["accuracy"] == 0.5
    assert m["recall"] == 0.5 and m["precision"] == 0.5 and m["specificity"] == 0.5
    assert m["auc"] == pytest.approx(0.75)


def test_bootstrap_ci_contains_accuracy():
    y, p = np.array([0, 1] * 50), np.array([0, 1] * 40 + [1, 0] * 10)
    lo, hi = training.bootstrap_ci(y, p)
    assert lo <= 0.8 <= hi and 0 <= lo < hi <= 1


def test_mcnemar():
    y = np.zeros(20, int)
    a = np.zeros(20, int)
    b = np.array([1] * 10 + [0] * 10)
    only_a, only_b, p = training.mcnemar(y, a, b)
    assert (only_a, only_b) == (10, 0) and p < 0.01
    assert training.mcnemar(y, a, a)[2] == 1.0


def test_model_can_overfit_a_small_batch(sets):
    """Sanity check of the training loop: 16 images must be learned perfectly."""
    torch.manual_seed(0)
    labels = sets["train"].labels
    subset = sets["train"].subset(np.r_[np.flatnonzero(labels == 0)[:8], np.flatnonzero(labels == 1)[:8]])
    model = build_model("cnn", SIZE)
    loader = data.make_loader(subset, batch_size=16, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(60):
        training.train_one_epoch(model, loader, optimizer, "cpu")
    probs, y, loss = training.predict(model, loader, "cpu")  # eval mode: no dropout
    assert (probs.argmax(1) == y).all() and loss < 0.1


def test_cross_validation_groups_never_split(sets, tmp_path, monkeypatch):
    """Copies of one X-ray (same group) must stay in the same fold."""
    groups = data.study_duplicates(sets)["groups"]
    seen = []
    original = training.make_loader

    def spy(image_set, indices=None, **kw):
        if indices is not None and not kw.get("shuffle"):
            seen.append(set(np.asarray(indices)))
        return original(image_set, indices, **kw)

    monkeypatch.setattr(training, "make_loader", spy)
    result = training.cross_validate("cnn", sets["train"], groups, "cpu", k_folds=3, epochs=1, batch_size=8,
                                     image_size=SIZE, results_dir=tmp_path)
    for val_idx in seen:
        for g in np.unique(groups):
            members = set(np.flatnonzero(groups == g))
            assert members <= val_idx or not (members & val_idx)
    assert len(result["folds"]) == 3 and len(result["histories"][0]["val_acc"]) == 1
    assert set(result["mean"]) == set(training.METRICS)
    saved = json.loads((tmp_path / "cv" / "cnn.json").read_text())
    assert saved["mean"]["accuracy"] == pytest.approx(result["mean"]["accuracy"])


def test_train_final_save_load_and_test(sets, tmp_path):
    model, info = training.train_final("cnn", sets["train"], sets["val"], "cpu", epochs=2, batch_size=8,
                                       image_size=SIZE, results_dir=tmp_path)
    assert 1 <= info["best_epoch"] <= 2 and info["parameters"] > 0
    loaded = training.load_model("cnn", tmp_path, "cpu", image_size=SIZE)
    x, _ = sets["test"].tensors([0, 1])
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-5)
    leaked = np.zeros(len(sets["test"]), bool)
    leaked[0] = True
    result = training.evaluate_test(loaded, sets["test"], "cpu", leaked=leaked)
    assert result["n_clean"] == len(sets["test"]) - 1
    assert np.array(result["confusion_matrix"]).sum() == len(sets["test"])
    assert len(result["y_pred"]) == len(sets["test"])


def test_tables_and_best_model():
    fake = {n: {"mean": {m: v for m in training.METRICS}, "std": {m: 0.01 for m in training.METRICS}}
            for n, v in [("cnn", 0.9), ("scatnet", 0.8)]}
    assert training.select_best(fake, "f1") == "cnn"
    assert list(training.cv_table(fake).index) == ["cnn", "scatnet"]
    test = {n: {"y_true": [0, 0, 1, 1], "y_pred": p} for n, p in [("cnn", [0, 0, 1, 1]), ("scatnet", [1, 0, 1, 0])]}
    errors = training.error_table(test, ["fractured", "not fractured"])
    assert errors.loc["scatnet", "missed fractured"] == 1 and errors.loc["scatnet", "false alarms"] == 1
    assert list(training.mcnemar_table(test).columns)[:3] == ["only A right", "only B right", "p-value"]
