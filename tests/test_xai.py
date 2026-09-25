import numpy as np
import pytest
import torch

from src import models, xai
from src.occlusion_scratch import occlusion_scratch
from tests.conftest import SIZE

FAST = dict(ig_steps=8, window=16, stride=8, lime_samples=40)


@pytest.fixture(scope="module")
def trained_like():
    """Models with random weights (enough to test the mechanics of the methods)."""
    torch.manual_seed(0)
    return {n: models.build_model(n, SIZE, pretrained=False).eval() for n in models.MODEL_NAMES}


@pytest.mark.parametrize("method", list(xai.METHODS))
@pytest.mark.parametrize("name", models.MODEL_NAMES)
def test_every_method_on_every_model(trained_like, images, method, name):
    x, y = images
    out = xai.METHODS[method](trained_like[name], x[:1], int(y[0]), **FAST)
    if method == "Grad-CAM" and name == "scatnet":
        assert out is None  # not applicable: no learned convolutional feature map
        return
    assert out.shape == (SIZE, SIZE) and np.isfinite(out).all()
    if method in ("Saliency", "Guided Backprop", "Grad-CAM"):
        assert (out >= 0).all()


def test_occlusion_from_scratch_matches_captum(trained_like, images):
    x, y = images
    for name, model in trained_like.items():
        ours = occlusion_scratch(model, x[:1], 1, window=16, stride=8, baseline=-1.0)
        captum = xai.occlusion(model, x[:1], 1, window=16, stride=8)
        np.testing.assert_allclose(ours, captum, atol=1e-5, rtol=1e-4, err_msg=name)


def test_occlusion_scratch_with_border_windows():
    """Window grid that does not fit exactly (last window clipped by the border), like Captum."""
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(40 * 40, 2)).eval()
    x = torch.randn(1, 1, 40, 40)
    np.testing.assert_allclose(occlusion_scratch(model, x, 0, window=16, stride=10, baseline=0.0),
                               xai.occlusion(model, x, 0, baseline=0.0, window=16, stride=10), atol=1e-5)


def test_integrated_gradients_completeness():
    """For a linear model, IG from a black image sums exactly to f(x) - f(black)."""
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(SIZE * SIZE, 2)).eval()
    x = torch.rand(1, 1, SIZE, SIZE) * 2 - 1
    attr = xai.integrated_gradients(model, x, 0, ig_steps=16)
    with torch.no_grad():
        delta = (model(x) - model(torch.full_like(x, -1.0)))[0, 0].item()
    assert attr.sum() == pytest.approx(delta, rel=1e-4, abs=1e-5)


def test_explain_all_cache_and_not_applicable(trained_like, images, tmp_path):
    x, y = images
    cache = tmp_path / "scatnet.npz"
    maps, seconds = xai.explain_all(trained_like["scatnet"], x, y, cache=cache, **FAST)
    assert set(maps) == set(xai.METHODS) and cache.exists()
    assert np.isnan(maps["Grad-CAM"]).all() and "Grad-CAM" not in xai.available(maps)
    again, seconds_again = xai.explain_all(trained_like["scatnet"], x, y, cache=cache)
    for k in maps:
        np.testing.assert_array_equal(np.nan_to_num(maps[k]), np.nan_to_num(again[k]))
    assert seconds_again == pytest.approx(seconds)


def test_deletion_curve_ends():
    """First point = the image untouched; last point = the whole image blacked out."""
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(SIZE * SIZE, 2)).eval()
    x = torch.rand(1, 1, SIZE, SIZE)
    curve = xai.deletion_curve(model, x, 0, np.random.rand(SIZE, SIZE), patch=16, steps=4)
    with torch.no_grad():
        assert curve[0] == pytest.approx(model(x).softmax(1)[0, 0].item(), abs=1e-6)
        assert curve[-1] == pytest.approx(model(torch.full_like(x, -1.0)).softmax(1)[0, 0].item(), abs=1e-6)


def test_deletion_rewards_the_right_map():
    """A model that only looks at the top-left patch: the map pointing there must beat the others."""
    class TopLeft(torch.nn.Module):
        def forward(self, x):
            s = x[:, 0, :16, :16].mean((1, 2)) * 10
            return torch.stack([s, -s], 1)

    x = torch.ones(1, 1, SIZE, SIZE)
    right = np.zeros((SIZE, SIZE))
    right[:16, :16] = 1
    wrong = np.zeros((SIZE, SIZE))
    wrong[-16:, -16:] = 1
    curves = xai.deletion_curves(TopLeft(), x, torch.tensor([0]), {"right": right[None], "wrong": wrong[None]},
                                 patch=16, steps=16)
    table = xai.deletion_table(curves)
    assert table.index[0] == "right"


def test_agreement_matrix_and_applicability(trained_like):
    maps = {"a": np.random.rand(2, SIZE, SIZE), "b": np.random.rand(2, SIZE, SIZE),
            "nan": np.full((2, SIZE, SIZE), np.nan)}
    maps["a_copy"] = maps["a"] * 3
    m = xai.agreement_matrix(maps)
    assert list(m.index) == ["a", "b", "a_copy"]
    assert m.loc["a", "a_copy"] == pytest.approx(1.0) and abs(m.loc["a", "b"]) < 0.5
    table = xai.applicability(trained_like)
    assert table.loc["Grad-CAM", "scatnet"].startswith("no")
    assert table.loc["Grad-CAM", "cnn"] == "yes"
