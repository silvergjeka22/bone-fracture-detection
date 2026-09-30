import numpy as np
import pytest

from src import localize

S = 64


def square(x0, y0, x1, y1, value=1.0):
    m = np.zeros((S, S))
    m[y0:y1, x0:x1] = value
    return m


def test_box_goes_around_the_blob_with_most_evidence():
    """A big blob and a small, brighter one: the box goes around the big one, not the brightest pixel."""
    m = square(10, 10, 30, 26) + square(45, 45, 50, 50, value=1.2)
    np.testing.assert_array_equal(localize.heatmap_to_box(m, threshold=0.3, smooth=0), [10, 10, 30, 26])


def test_only_positive_evidence_counts():
    m = square(5, 5, 20, 20, value=-3.0) + square(40, 40, 50, 50)
    np.testing.assert_array_equal(localize.heatmap_to_box(m, smooth=0), [40, 40, 50, 50])
    assert localize.heatmap_to_box(-np.ones((S, S))) is None
    assert localize.heatmap_to_box(np.full((S, S), np.nan)) is None  # method not applicable


def test_iou():
    box = np.array([0, 0, 10, 10.0])
    assert localize.iou(box, [[0, 0, 10, 10]]) == pytest.approx(1)
    assert localize.iou(box, [[20, 20, 30, 30]]) == 0
    assert localize.iou(box, [[5, 0, 15, 10], [20, 20, 30, 30]]) == pytest.approx(50 / 150)  # best match
    assert localize.iou(None, [[0, 0, 10, 10]]) == 0


def test_pointing_game():
    truth = np.array([[10, 10, 20, 20]], dtype=float)
    assert localize.pointing_game(square(12, 12, 16, 16), truth, smooth=0)
    assert not localize.pointing_game(square(40, 40, 44, 44), truth, smooth=0)
    assert not localize.pointing_game(np.zeros((S, S)), truth)


def test_localization_table():
    truth = [np.array([[10, 10, 20, 20]], dtype=float)] * 2
    maps = {"Occlusion": np.stack([square(11, 11, 19, 19)] * 2),
            "Saliency": np.stack([square(40, 40, 50, 50)] * 2),
            "Grad-CAM": np.full((2, S, S), np.nan)}  # not applicable: left out
    table = localize.localization_table(maps, truth, smooth=0)
    assert list(table.index) == ["Occlusion", "Random", "Saliency"]  # sorted by hit rate
    assert table.loc["Occlusion", "hit rate (%)"] == 100 and table.loc["Saliency", "hit rate (%)"] == 0
    assert table.loc["Occlusion", "IoU"] == pytest.approx(64 / 100)
    assert table.loc["Random", "hit rate (%)"] == pytest.approx(100 * 100 / S ** 2, abs=0.1)


def test_box_lands_on_the_synthetic_crack(dataset):
    """A map that is hot on the crack of a synthetic X-ray gives a box on the radiologist's box."""
    i = int(np.flatnonzero(dataset.labels == 0)[0])
    x0, y0, x1, y1 = dataset.boxes[i][0].round().astype(int)
    m = np.zeros(dataset.images[i].shape)
    m[y0:y1, x0:x1] = 1.0
    box = localize.heatmap_to_box(m)
    assert localize.iou(box, dataset.boxes[i]) > 0.5 and localize.pointing_game(m, dataset.boxes[i])
