from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import default_collate

from src import joint, models, training
from tests.conftest import SIZE


def batch(image_set, n=6):
    """n fractured + n healthy X-rays as one batch."""
    idx = [*np.flatnonzero(image_set.labels == 0)[:n], *np.flatnonzero(image_set.labels == 1)[:n]]
    ds = joint.BoxDataset(image_set)
    return default_collate([ds[i] for i in idx])


def test_targets_peak_at_the_box_centre():
    heat, reg, centre = joint.targets(torch.tensor([[16.0, 32.0, 48.0, 48.0]]), SIZE)  # centre (32, 40) px: cell (2, 2)
    assert heat.shape == (SIZE // 16, SIZE // 16) and heat[2, 2] == 1 and centre.sum() == 1 and centre[2, 2]
    assert torch.allclose(reg[:, 2, 2], torch.tensor([0.0, 0.5, 2.0, 1.0]))
    assert joint.targets(torch.zeros(0, 4), SIZE)[0].sum() == 0  # healthy: nothing to find


def test_boxes_survive_the_augmentation_path():
    masks = torch.zeros(1, SIZE, SIZE)
    masks[0, 10:30, 5:20] = 1
    assert torch.equal(joint.boxes_from_masks(masks), torch.tensor([[5.0, 10.0, 20.0, 30.0]]))
    assert joint.boxes_from_masks(torch.zeros(1, SIZE, SIZE)).shape == (0, 4)


def test_box_dataset(sets):
    ds = joint.BoxDataset(sets["train"], margin=0)
    for i, (label, boxes) in enumerate(zip(sets["train"].labels, sets["train"].boxes)):
        x, y, mask, heat, reg, centre = ds[i]
        assert x.shape == mask.shape == (1, SIZE, SIZE) and y == label
        assert (mask.sum() > 0) == (len(boxes) > 0) == bool(centre.any())
    x, _, mask, *_ = joint.BoxDataset(sets["train"], augment=True)[0]
    assert x.shape == mask.shape


def test_blur_and_moved_patch():
    x = torch.randn(1, 1, SIZE, SIZE)
    mask = torch.zeros(1, 1, SIZE, SIZE)
    mask[..., 4:12, 4:12] = 1
    out = joint.blur_inside(x, mask)
    assert torch.equal(out[mask == 0], x[mask == 0]) and not torch.equal(out[mask == 1], x[mask == 1])
    other = joint.moved(mask)
    assert other.sum() == mask.sum() and (other * mask).sum() == 0


def test_xai_losses_teach_classifier_and_detector(sets):
    model = models.build_model("joint", SIZE, pretrained=False).train()
    x, y, mask, heat, reg, centre = batch(sets["train"])
    _, losses = joint.joint_losses(model, x, y, mask, heat, reg, centre)
    assert all(torch.isfinite(v) for v in losses.values()) and losses["point"] > 0 and losses["erase"] > 0
    (losses["point"] + losses["agree"] + losses["erase"]).backward()  # the XAI losses alone ...
    grads = dict(model.named_parameters())
    assert grads["detector.0.weight"].grad.abs().sum() > 0           # ... move the detector (agreement)
    assert grads["classifier.1.weight"].grad.abs().sum() > 0         # ... and the classifier
    _, off = joint.joint_losses(model, x, y, mask, heat, reg, centre, detect=False, point=False, agree=False, erase=False)
    assert all(off[k] == 0 for k in ("detect", "point", "agree", "erase"))


def test_find_boxes_decodes_a_peak():
    g = SIZE // 16
    out = torch.full((1, 5, g, g), -10.0)
    out[0, :, 1, 2] = torch.tensor([5.0, 0.0, 0.0, 2.0, 1.0])  # centre in cell (1, 2), offset 0.5, 32 x 16 px
    fake = SimpleNamespace(eval=lambda: None, grid=lambda x: x, detector=lambda a: out)
    (found,) = joint.find_boxes(fake, torch.zeros(1, 1, SIZE, SIZE), "cpu")
    np.testing.assert_allclose(found[:, :4], [[24, 16, 56, 32]], atol=1e-4)
    assert found[0, 4] == pytest.approx(torch.sigmoid(torch.tensor(5.0)).item())


def test_train_joint_warm_up_save_and_reload(sets, tmp_path):
    model, info = joint.train_joint("joint_xai", sets["train"], sets["val"], "cpu", detect=1, point=0.5, agree=0.5,
                                    erase=0.5, warmup=1, epochs=2, batch_size=8, num_workers=0, results_dir=tmp_path)
    h = info["history"]
    assert h["point"][0] == 0 and h["point"][1] > 0 and h["detect"][0] > 0  # XAI losses only after the warm-up
    assert 0.05 <= info["threshold"] <= 0.95
    loaded = training.load_model("joint_xai", tmp_path, "cpu", arch="joint")
    x = sets["val"].tensors([0, 1])[0]
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-5)


def test_erase_test_and_detector_scores(sets):
    model = models.build_model("joint", SIZE, pretrained=False).eval()
    erase = joint.erase_test(model, sets["test"], "cpu")
    assert list(erase) == ["as they are", "fracture blurred", "random patch blurred"]
    assert all(0 <= v <= 100 for v in erase.values())
    frac = np.flatnonzero(sets["test"].labels == 0)
    truth = [sets["test"].boxes[i] for i in frac]
    perfect = [np.column_stack([t, np.ones(len(t))]).astype(np.float32) for t in truth]
    assert joint.detector_scores(perfect, truth, SIZE)["AP@0.5 (%)"] == 100
    found = joint.find_boxes(model, sets["test"].tensors(frac)[0], "cpu")
    assert len(found) == len(frac) and all(f.shape[1] == 5 for f in found)


def test_best_on_val_and_comparison():
    final = {"a": {"history": {"val_auc": [0.7, 0.8]}, "best_epoch": 2},
             "c": {"history": {"val_auc": [0.9, 0.6]}, "best_epoch": 1}}
    assert joint.best_on_val(final, ["a", "c"]) == "c"
    test = {"c": {"accuracy": 0.9, "f1": 0.7, "recall": 0.6, "auc": 0.8}}
    hits = pd.DataFrame({"hit rate (%)": {"Saliency": 20.0, "Grad-CAM": 40.0, "Random": 3.0}})
    deletion = pd.DataFrame({"deletion AUC (mean)": {"Saliency": 0.3, "Grad-CAM": 0.5, "Random": 0.6}})
    erase = {"as they are": 80.0, "fracture blurred": 30.0, "random patch blurred": 75.0}
    row = joint.comparison(["c"], test, {"c": hits}, {"c": deletion}, {"c": {"AP@0.5 (%)": 50.0, "hit rate (%)": 60.0}},
                           {"c": 45.0}, {"c": erase}, {"c": "C"}).loc["C"]
    assert row["best XAI method"] == "Grad-CAM" and row["XAI hit rate, mean of 6 (%)"] == 30
    assert row["p(fractured) drop, fracture blurred"] == 50 and row["p(fractured) drop, random patch blurred"] == 5
