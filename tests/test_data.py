import numpy as np
import pytest
import torch

from src import data
from tests.conftest import SIZE


def test_find_data_dir(data_dir):
    assert data.find_data_dir(data_dir.parent) == data_dir
    with pytest.raises(FileNotFoundError):
        data.find_data_dir(data_dir / "train")


def test_load_split(sets):
    train = sets["train"]
    assert train.class_names == ["fractured", "not fractured"]
    assert train.images.shape == (len(train), SIZE, SIZE) and train.images.dtype == np.uint8
    assert set(np.unique(train.labels)) == {0, 1}
    assert train.sizes.shape == (len(train), 2) and (train.sizes > 0).all()
    assert len(train) == 12 * 2 + 4  # 12 per class + 2 flipped/rotated copies per class


def test_cache_roundtrip(data_dir, tmp_path):
    first = data.load_split(data_dir / "val", size=SIZE, cache_dir=tmp_path)
    assert list(tmp_path.glob("*.npz"))
    second = data.load_split(data_dir / "val", size=SIZE, cache_dir=tmp_path)
    np.testing.assert_array_equal(first.images, second.images)
    assert first.paths == second.paths and first.class_names == second.class_names


def test_max_per_class(data_dir):
    small = data.load_split(data_dir / "train", size=SIZE, max_per_class=3)
    assert np.bincount(small.labels).tolist() == [3, 3]


def test_unreadable_file_is_skipped(data_dir, tmp_path):
    for name in ("fractured", "not fractured"):
        (tmp_path / "split" / name).mkdir(parents=True)
        (tmp_path / "split" / name / "broken.png").write_bytes(b"not an image")
        (tmp_path / "split" / name / "ok.png").write_bytes((data_dir / "val" / name / next(
            p.name for p in (data_dir / "val" / name).iterdir())).read_bytes())
    loaded = data.load_split(tmp_path / "split", size=SIZE)
    assert len(loaded) == 2


def test_dataset_and_loader(sets):
    loader = data.make_loader(sets["train"], augment=True, batch_size=5, shuffle=True)
    x, y = next(iter(loader))
    assert x.shape == (5, 1, SIZE, SIZE) and x.dtype == torch.float32
    assert x.min() >= -1 and x.max() <= 1 and y.dtype == torch.int64
    x, y = sets["test"].tensors([0, 1])
    assert x.shape == (2, 1, SIZE, SIZE)
    assert np.allclose(data.denormalize(x[0]), sets["test"].images[0] / 255, atol=1e-6)


def test_near_duplicates_finds_flips_rotations_and_brightness(sets):
    study = data.study_duplicates(sets, threshold=0.97)
    train = sets["train"]
    names = [p.split("/")[-1] for p in train.paths]
    # the flipped and rotated copies share a group with their original, the other images are alone
    for label in (0, 1):
        members = [names.index(f"train_{label}_{s}.png") for s in ("000", "dup_flip", "dup_rot")]
        assert len({study["groups"][m] for m in members}) == 1
    sizes = np.bincount(study["groups"])
    assert (sizes > 1).sum() == 2 and sizes.max() == 3
    # exactly the two brightened test copies leak from train
    leaked = [p.split("/")[-1] for p, flag in zip(sets["test"].paths, study["leaked"]["test"]) if flag]
    assert sorted(leaked) == ["test_0_leak.jpg", "test_1_leak.jpg"]
    assert not study["leaked"]["val"].any()


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
