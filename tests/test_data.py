import shutil

import numpy as np
import pytest
import torch

from src import data
from tests.conftest import SIZE


def test_find_data_dir(data_dir):
    assert data.find_data_dir(data_dir.parent) == data_dir
    with pytest.raises(FileNotFoundError):
        data.find_data_dir(data_dir / "Annotations")


def test_load_dataset(dataset):
    assert dataset.class_names == ["fractured", "not fractured"]
    assert dataset.images.shape == (len(dataset), SIZE, SIZE) and dataset.images.dtype == np.uint8
    assert np.bincount(dataset.labels).tolist() == [16 + 1, 16 + 2]  # + the planted copies
    assert dataset.sizes.shape == (len(dataset), 2) and (dataset.sizes > 0).all()


def test_fracture_boxes(dataset):
    """Every fractured X-ray has its box, rescaled to the image; healthy ones have none."""
    for label, boxes in zip(dataset.labels, dataset.boxes):
        assert boxes.shape == ((1, 4) if label == 0 else (0, 4))
        if label == 0:
            x0, y0, x1, y1 = boxes[0]
            assert 0 <= x0 < x1 <= SIZE and 0 <= y0 < y1 <= SIZE
    names = [p.split("/")[-1] for p in dataset.paths]
    np.testing.assert_array_equal(dataset.boxes[names.index("IMG0000001.jpg")],
                                  dataset.boxes[names.index("IMG0000001_bright.jpg")])


def test_read_boxes_yolo(tmp_path):
    (tmp_path / "a.txt").write_text("0 0.5 0.25 0.2 0.1\n0 0.1 0.1 0.4 0.4\n")
    boxes = data.read_boxes(tmp_path / "a.txt", 100)
    np.testing.assert_allclose(boxes, [[40, 20, 60, 30], [0, 0, 30, 30]], atol=1e-4)  # clipped at the border
    assert data.read_boxes(tmp_path / "missing.txt", 100).shape == (0, 4)


def test_cache_roundtrip(data_dir, tmp_path):
    first = data.load_dataset(data_dir, size=SIZE, cache_dir=tmp_path)
    assert list(tmp_path.glob("*.npz"))
    second = data.load_dataset(data_dir, size=SIZE, cache_dir=tmp_path)
    np.testing.assert_array_equal(first.images, second.images)
    assert first.paths == second.paths
    assert all(np.array_equal(a, b) for a, b in zip(first.boxes, second.boxes))


def test_max_per_class(data_dir):
    small = data.load_dataset(data_dir, size=SIZE, max_per_class=3)
    assert np.bincount(small.labels).tolist() == [3, 3]


def test_unreadable_file_is_skipped(data_dir, tmp_path):
    for folder in data.CLASS_FOLDERS:
        (tmp_path / "images" / folder).mkdir(parents=True)
        (tmp_path / "images" / folder / "broken.jpg").write_bytes(b"not an image")
        shutil.copy(next((data_dir / "images" / folder).iterdir()), tmp_path / "images" / folder / "ok.jpg")
    (tmp_path / "Annotations" / "YOLO").mkdir(parents=True)
    assert len(data.load_dataset(tmp_path, size=SIZE)) == 2


def test_dataset_and_loader(sets):
    loader = data.make_loader(sets["train"], augment=True, batch_size=5, shuffle=True)
    x, y = next(iter(loader))
    assert x.shape == (5, 1, SIZE, SIZE) and x.dtype == torch.float32
    assert x.min() >= -1 and x.max() <= 1 and y.dtype == torch.int64
    x, y = sets["test"].tensors([0, 1])
    assert x.shape == (2, 1, SIZE, SIZE)
    assert np.allclose(data.denormalize(x[0]), sets["test"].images[0] / 255, atol=1e-6)


def test_near_duplicates_finds_flips_rotations_and_brightness(dataset, dup):
    names = [p.split("/")[-1] for p in dataset.paths]
    planted = (["IMG0000017.jpg", "IMG0000017_flip.jpg", "IMG0000017_rot.jpg"],  # first healthy X-ray
               ["IMG0000001.jpg", "IMG0000001_bright.jpg"])                      # first fractured X-ray
    for members in planted:
        assert len({dup["groups"][names.index(m)] for m in members}) == 1
    sizes = np.bincount(dup["groups"])
    assert (sizes > 1).sum() == 2 and sizes.max() == 3  # the other images are alone
    assert dup["table"].loc["dataset", "with a near-duplicate"] == 5


def test_split_is_stratified_grouped_and_complete(dataset, dup, split):
    parts = [split[name] for name in ("train", "val", "test")]
    assert sorted(np.concatenate(parts).tolist()) == list(range(len(dataset)))  # every image exactly once
    for g in np.unique(dup["groups"]):  # the copies of one X-ray stay in one part
        members = np.flatnonzero(dup["groups"] == g)
        assert sum(np.isin(members, p).any() for p in parts) == 1
    for p in parts:
        assert set(dataset.labels[p].tolist()) == {0, 1}
    assert len(split["train"]) > len(split["val"]) and len(split["train"]) > len(split["test"])


def test_pick_images_balanced_and_falls_back():
    labels = np.array([0] * 6 + [1] * 6)
    allowed = np.array([True] * 6 + [False] * 5 + [True])
    idx = data.pick_images(labels, 2, allowed=allowed)
    assert labels[idx].tolist() == [0, 0, 1, 1]
    assert 11 in idx  # the only allowed image of class 1 is used first


def test_summary_table(sets):
    table = data.summary_table(sets)
    assert list(table.index) == ["train", "val", "test"]
    assert table.loc["train", "total"] == len(sets["train"])
    assert table.loc["train", "fracture boxes"] == (sets["train"].labels == 0).sum()
