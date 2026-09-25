import pytest
import torch

from src import models
from tests.conftest import SIZE


@pytest.mark.parametrize("name", models.MODEL_NAMES)
def test_forward_shape(name):
    model = models.build_model(name, image_size=SIZE, pretrained=False).eval()
    assert model(torch.randn(3, 1, SIZE, SIZE)).shape == (3, 2)


def test_full_size_classifier_inputs():
    """At 224x224 both CNN and ScatNet end on a 14x14 grid (exam: only the input size differs)."""
    assert models.build_model("cnn").classifier[1].in_features == 256 * 14 * 14
    assert models.build_model("scatnet").classifier[1].in_features == 417 * 14 * 14


def test_same_classifier_for_every_model():
    layouts = [models.classifier_layout(models.build_model(n, image_size=SIZE, pretrained=False))
               for n in models.MODEL_NAMES]
    assert all(layout == layouts[0] for layout in layouts)
    assert all(type(m.classifier) is models.Classifier
               for m in (models.build_model(n, SIZE, pretrained=False) for n in models.MODEL_NAMES))


def test_scatnet_learns_no_filters():
    model = models.ScatNet(image_size=SIZE)
    assert not any(p.requires_grad for p in model.scattering.parameters())
    feature_params = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("classifier"))
    assert feature_params == 2 * model.n_coefficients  # only the batch-norm scale and shift
    assert model.n_coefficients == 1 + 4 * 8 + 8 * 8 * 4 * 3 // 2 == 417


def test_scatnet_features_finite_on_black_image():
    model = models.ScatNet(image_size=SIZE).eval()
    assert torch.isfinite(model(torch.full((2, 1, SIZE, SIZE), -1.0))).all()


def test_gradients_reach_the_input():
    for name in models.MODEL_NAMES:
        model = models.build_model(name, SIZE, pretrained=False).eval()
        x = torch.randn(1, 1, SIZE, SIZE, requires_grad=True)
        model(x)[0, 0].backward()
        assert x.grad is not None and x.grad.abs().sum() > 0, name


def test_cam_layers():
    assert models.build_model("cnn", SIZE).cam_layer is not None
    assert models.build_model("resnet18", SIZE, pretrained=False).cam_layer is not None
    assert getattr(models.build_model("scatnet", SIZE), "cam_layer", None) is None


def test_cnn_first_layer_is_7x7_grey():
    assert tuple(models.build_model("cnn").first_conv.weight.shape) == (32, 1, 7, 7)


def test_resnet_relus_not_inplace():
    model = models.build_model("resnet18", SIZE, pretrained=False)
    assert not any(m.inplace for m in model.modules() if isinstance(m, torch.nn.ReLU))


def test_unknown_model():
    with pytest.raises(ValueError):
        models.build_model("vgg")


def test_summary_table():
    table = models.summary(["cnn", "scatnet"], image_size=SIZE)
    assert list(table.index) == ["cnn", "scatnet"]
