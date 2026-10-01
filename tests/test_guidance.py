import pandas as pd
import pytest
import torch

from src import guidance, models, training
from tests.conftest import SIZE


def batch(image_set, n=12):
    items = [guidance.BoxDataset(image_set)[i] for i in range(n)]
    return (torch.stack([i[0] for i in items]), torch.tensor([i[1] for i in items]), torch.stack([i[2] for i in items]))


def test_box_dataset_masks(sets):
    ds = guidance.BoxDataset(sets["train"], margin=2)
    for i, (label, boxes) in enumerate(zip(sets["train"].labels, sets["train"].boxes)):
        x, y, mask = ds[i]
        assert x.shape == mask.shape == (1, SIZE, SIZE) and y == label
        if len(boxes):
            x0, y0, x1, y1 = boxes[0].astype(int)
            assert mask[0, y0:y1, x0:x1].min() == 1 and mask.sum() < SIZE * SIZE
        else:
            assert mask.sum() == 0


def test_guided_losses_push_the_model(sets):
    model = models.build_model("cnn", SIZE).train()
    head = torch.nn.Linear(guidance.CHANNELS, 16)
    x, y, mask = batch(sets["train"])
    logits, energy, con = guidance.guided_losses(model, head, x, y, mask)
    assert logits.shape == (len(y), 2) and 0 <= energy.item() <= 1 and torch.isfinite(con)
    (energy + con).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    _, off_energy, off_con = guidance.guided_losses(model, head, x, y, mask, box=False, contrast=False)
    assert off_energy.item() == 0 and off_con.item() == 0


def test_contrastive_loss_rewards_alike_fractures():
    """Fracture regions that look alike (and unlike the background) give a lower loss."""
    mask = torch.zeros(4, 2, 2)
    mask[:2, 0, 0] = 1  # X-rays 0 and 1 are fractured, box on the top-left cell
    fractured = torch.tensor([True, True, False, False])
    features = torch.zeros(4, 2, 2, 2)
    features[:, 1] = 1.0                                        # background: direction (0, 1)
    same = guidance.contrastive_loss(torch.nn.Identity(), features, mask, fractured)
    features[:2, :, 0, 0] = torch.tensor([1.0, 0.0])[None, :]   # fracture cells: direction (1, 0)
    separated = guidance.contrastive_loss(torch.nn.Identity(), features, mask, fractured)
    assert separated < same


def test_train_guided_warm_up_then_guide(sets, tmp_path):
    model, info = guidance.train_guided("cnn", "cnn_box_contrast", sets["train"], sets["val"], "cpu", warmup=1,
                                        epochs=2, batch_size=8, num_workers=0, image_size=SIZE, results_dir=tmp_path)
    assert info["history"]["energy"][0] == 0 and info["history"]["energy"][1] > 0  # off in the warm-up
    loaded = training.load_model("cnn_box_contrast", tmp_path, "cpu", image_size=SIZE, arch="cnn")
    x, _, _ = batch(sets["val"], n=2)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-5)


def test_best_on_val_and_comparison():
    final = {"a": {"history": {"val_f1": [0.1, 0.5]}, "best_epoch": 2},
             "b": {"history": {"val_f1": [0.6, 0.2]}, "best_epoch": 1}}
    assert guidance.best_on_val(final, ["a", "b"]) == "b"
    test = {"m": {"accuracy": 0.9, "f1": 0.7, "recall": 0.6, "auc": 0.8}}
    hits = pd.DataFrame({"hit rate (%)": {"Saliency": 20.0, "Grad-CAM": 40.0, "Random": 3.0}})
    deletion = pd.DataFrame({"deletion AUC (mean)": {"Saliency": 0.3, "Grad-CAM": 0.5, "Random": 0.6}})
    row = guidance.comparison(["m"], test, {"m": hits}, {"m": deletion}, {"m": "M"}).loc["M"]
    assert row["best method"] == "Grad-CAM" and row["hit rate, mean of methods (%)"] == 30
    assert row["deletion AUC, mean (lower = better)"] == pytest.approx(0.4)
