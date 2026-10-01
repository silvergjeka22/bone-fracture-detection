import numpy as np
import pytest
import torch

from src import pipeline, xai


def test_verdict_needs_agreement():
    assert pipeline.verdict(0.9, has_box=True) == "fracture"
    assert pipeline.verdict(0.1, has_box=False) == "no fracture"
    assert pipeline.verdict(0.9, has_box=False) == "needs review"   # classifier yes, detector no
    assert pipeline.verdict(0.1, has_box=True) == "needs review"    # detector yes, classifier no


def test_run_reports_and_scores(sets):
    test = sets["test"]
    n, size = len(test), test.images.shape[1]
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(size * size, 2)).eval()
    probs = np.where(np.arange(n) % 2 == 0, 0.9, 0.1)                     # even X-rays called fractured
    box = np.array([[0, 0, size, size, 0.7]], dtype=np.float32)          # covers the whole image
    detections = [box if k % 4 < 2 else np.zeros((0, 5), np.float32) for k in range(n)]
    cases = pipeline.run(model, test, probs, detections, "Saliency", "cpu")

    expected = [pipeline.verdict(p, len(d) > 0) for p, d in zip(probs, detections)]
    assert cases["verdict"].tolist() == expected
    explained = cases["p(fractured)"] >= 0.5
    assert cases.loc[explained, "xai box"].notna().all() and cases.loc[~explained, "xai box"].isna().all()
    with_box = explained & cases["detector box"].notna()
    assert cases.loc[with_box, "explanation agrees"].astype(bool).all()  # the box covers the whole image
    assert "needs a radiologist's review" in cases["report"][cases["verdict"] == "needs review"].iloc[0]
    assert "detector: fracture at x 0-" in cases["report"][0]

    table = pipeline.outcomes(cases)
    assert list(table.columns) == list(pipeline.VERDICTS) and table.values.sum() == n
    scores = pipeline.scores(cases)
    assert scores["needs review (%)"] == pytest.approx(100 * (cases["verdict"] == "needs review").mean())
    assert scores["explanation inside the detector box (%)"] == 100


def test_report_uses_original_pixels():
    row = {"image": "IMG1.jpg", "p(fractured)": 0.8, "detector box": np.array([10, 20, 30, 40.0]),
           "detector confidence": 0.5, "explanation agrees": False, "verdict": "fracture"}
    text = pipeline.write_report(row, original_size=(448, 224), size=224, method="Occlusion")
    assert "x 20-60, y 20-40 px" in text and "outside the detected box" in text and text.startswith("IMG1.jpg: FRACTURE")


def test_pick_method_on_the_val_xrays(sets):
    size = sets["val"].images.shape[1]
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(size * size, 2)).eval()
    method, table = pipeline.pick_method(model, sets["val"], ig_steps=4, window=16, stride=16, lime_samples=20)
    assert method in xai.METHODS and "Random" in table.index and method != "Grad-CAM"  # no conv layer here
