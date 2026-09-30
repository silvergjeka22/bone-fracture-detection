"""The YOLO glue code, with a fake `ultralytics` (the real training runs on Kaggle and in CI's notebook test)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src import detect
from tests.conftest import SIZE


class FakeYOLO:
    """Stands in for ultralytics.YOLO: same calls, predictable outputs."""

    def __init__(self, model):
        self.model, self.trainer = str(model), None

    def train(self, data, project, name, **kwargs):
        assert Path(data).exists()
        weights = Path(project) / name / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        (weights / "best.pt").write_bytes(b"weights")
        self.trainer = SimpleNamespace(best=weights / "best.pt")

    def val(self, data, split, **kwargs):
        return SimpleNamespace(box=SimpleNamespace(map50=0.5, map=0.25, mp=0.6, mr=0.4))

    def predict(self, source, conf, **kwargs):
        for k, _ in enumerate(source):  # image k gets k % 3 boxes
            boxes = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.9, 0.9]])[: k % 3]
            yield SimpleNamespace(boxes=SimpleNamespace(xyxyn=boxes, conf=torch.tensor([0.4, 0.9])[: k % 3]))


@pytest.fixture
def fake_ultralytics(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))


def test_write_yolo_dataset(sets, tmp_path):
    data_yaml = detect.write_yolo_dataset(sets, tmp_path / "yolo")
    text = data_yaml.read_text()
    assert "train: images/train" in text and "test: images/test" in text and "0: fracture" in text
    for split, image_set in sets.items():
        fractured = [Path(p) for p, b in zip(image_set.paths, image_set.boxes) if len(b)]
        assert sorted(p.name for p in (tmp_path / "yolo" / "images" / split).iterdir()) == sorted(p.name for p in fractured)
        i = next(k for k, b in enumerate(image_set.boxes) if len(b))
        label = np.loadtxt(tmp_path / "yolo" / "labels" / split / f"{Path(image_set.paths[i]).stem}.txt", ndmin=2)
        x0, y0, x1, y1 = image_set.boxes[i][0] / SIZE
        np.testing.assert_allclose(label[0], [0, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], atol=1e-5)


def test_to_boxes_scales_and_sorts():
    boxes = detect.to_boxes([[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 1.0, 1.0]], [0.3, 0.8], size=100)
    np.testing.assert_allclose(boxes, [[50, 50, 100, 100, 0.8], [10, 20, 30, 40, 0.3]], atol=1e-5)
    assert detect.to_boxes(np.zeros((0, 4)), np.zeros(0), 100).shape == (0, 5)


def test_train_evaluate_predict_glue(fake_ultralytics, sets, tmp_path):
    data_yaml = detect.write_yolo_dataset(sets, tmp_path / "yolo")
    weights = detect.train(data_yaml, tmp_path / "detector", model="yolov8n.yaml", epochs=1)
    assert weights == (tmp_path / "detector" / "best.pt").resolve() and weights.exists()
    assert detect.load(tmp_path / "detector").exists()
    assert detect.box_metrics(weights, data_yaml) == {"mAP@0.5": 0.5, "mAP@0.5:0.95": 0.25, "precision": 0.6, "recall": 0.4}
    found = detect.find_boxes(weights, ["a.jpg", "b.jpg", "c.jpg"], size=SIZE)
    assert [len(b) for b in found] == [0, 1, 2]
    assert found[2][0, 4] == pytest.approx(0.9) and found[2][0, 0] == pytest.approx(0.5 * SIZE)  # most confident first


def test_load_without_training(tmp_path):
    with pytest.raises(FileNotFoundError):
        detect.load(tmp_path)
