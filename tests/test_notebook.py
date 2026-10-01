"""Run the whole notebook top to bottom on synthetic data (QUICK mode): the proof that it works end to end.

Slow (~5-10 minutes on a CPU): `pytest -m "not slow"` skips it.
ResNet18's ImageNet weights are replaced by random ones saved in a temporary TORCH_HOME, and QUICK mode
builds YOLO from its .yaml (no pretrained download), so the test needs no internet. It needs `ultralytics`.
"""

import json
import os
from pathlib import Path

import nbformat
import pytest
import torch
import torchvision
from nbclient import NotebookClient

from tests.synthetic import make_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_notebook_runs_end_to_end(tmp_path):
    data_dir = make_dataset(tmp_path / "data", per_class=30)
    results = tmp_path / "results"
    torch_home = tmp_path / "torch"
    url = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.url
    (torch_home / "hub" / "checkpoints").mkdir(parents=True)
    torch.save(torchvision.models.resnet18().state_dict(), torch_home / "hub" / "checkpoints" / url.split("/")[-1])

    nb = nbformat.read(ROOT / "notebooks" / "main.ipynb", as_version=4)
    settings = [c for c in nb.cells if c.cell_type == "code" and c.source.startswith("QUICK")]
    assert len(settings) == 1
    settings[0].source = settings[0].source.replace("QUICK      = False", "QUICK      = True")
    assert "QUICK      = True" in settings[0].source
    settings[0].source = settings[0].source.replace("JOINT_SIZE = 448", "JOINT_SIZE = 224")  # CPU: smaller X-rays

    env = {"BFD_DATA_DIR": str(data_dir), "BFD_RESULTS_DIR": str(results), "TORCH_HOME": str(torch_home)}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        NotebookClient(nb, timeout=1800, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    summary = json.loads((results / "summary.json").read_text())
    assert set(summary["models"]) == {"cnn", "scatnet", "resnet18", "separate", "joint", "joint_xai"}
    assert summary["best_model"] in summary["models"] and len(summary["joint"]["comparison"]) == 3
    assert "Grad-CAM" in summary["joint"]["hit_tests"] and summary["pipeline"]["classifier"] in summary["models"]
    for figure in ("dataset_overview", "samples_train", "duplicates", "learning_curves", "confusion_matrices",
                   "filters_cnn", "filters_scatnet", "xai_cnn", "xai_scatnet", "occlusion_scratch_vs_captum",
                   "deletion_curves", "xai_agreement", "boxes_cnn", "boxes_scatnet", "localization",
                   "pipeline_examples", "joint_training", "joint_localization", "erase_test", "xai_joint_xai",
                   "boxes_joint_xai", "joint_examples"):
        assert (results / "figures" / f"{figure}.png").exists(), figure
    assert max(v["max |difference|"] for v in summary["scratch_vs_captum"].values()) < 1e-4
    assert {"cnn", "scatnet"} <= set(summary["localization"]) and "Random" in summary["localization"]["cnn"]
    assert set(summary["pipeline"]) == {"classifier", "detector", "detector_boxes", "cases", "box_method"}
    assert (results / "detector" / "best.pt").exists()
