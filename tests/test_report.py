import pandas as pd

from src import report


def parts():
    metrics = {"accuracy": 0.9, "f1": 0.7, "precision": 0.7, "recall": 0.7, "specificity": 0.95, "auc": 0.9}
    test = {n: {**metrics, "accuracy_ci": [0.87, 0.93], "confusion_matrix": [[1, 0], [0, 1]], "y_pred": [0]}
            for n in ("cnn", "joint_xai")}
    final = {n: {"parameters": 1000, "best_epoch": 3} for n in test}
    deletion = {"cnn": pd.DataFrame({"deletion AUC (mean)": {"Occlusion": 0.2, "Random": 0.5}})}
    hits = pd.DataFrame({"hit rate (%)": {"Occlusion": 40.0, "Random": 3.0}, "IoU": {"Occlusion": 0.2, "Random": None}})
    comparison = pd.DataFrame({"test F1 (%)": [70.0], "fractures found (%)": [75.0], "detector AP@0.5 (%)": [50.0],
                               "detector hit rate (%)": [60.0], "Grad-CAM hit rate (%)": [40.0],
                               "best XAI method": ["Occlusion"], "best XAI hit rate (%)": [45.0],
                               "p(fractured) drop, fracture blurred": [30.0]}, index=["C: joint + XAI"])
    hit_tests = pd.DataFrame({"hits A": [10], "hits C": [20], "p-value": [0.01]}, index=["Grad-CAM"])
    return dict(settings={"xai_per_class": 5}, dataset=pd.DataFrame({"total": [10]}, index=["train"]),
                duplicates=pd.DataFrame({"images": [10]}, index=["dataset"]),
                cv={"cnn": {"mean": metrics, "std": metrics}}, final=final, test=test, best="cnn", select_by="f1",
                mcnemar=pd.DataFrame({"p-value": [0.03]}, index=["cnn vs resnet18"]), deletion=deletion,
                seconds={"cnn": {"Occlusion": 0.5}}, scratch=pd.DataFrame({"max |difference|": [1e-7]}, index=["cnn"]),
                localization={"cnn": hits}, joint=dict(comparison=comparison, hit_tests=hit_tests),
                pipeline={"classifier": "joint_xai", "box_method": "Occlusion",
                          "detector": {"mAP@0.5": 0.58}, "detector_boxes": {"hit rate (%)": 70.0},
                          "cases": {"needs review (%)": 12.0}})


def test_export_writes_summary_and_findings(tmp_path):
    summary = report.export(tmp_path, **parts())
    assert (tmp_path / "summary.json").exists()
    assert summary["models"]["joint_xai"]["cv_mean"] is None  # the joint versions have no cross-validation
    assert summary["xai"]["cnn"]["deletion_auc"]["Occlusion"] == 0.2
    text = report.key_findings(summary)
    assert "C: joint + XAI" in text and "mAP@0.5 58.0%" in text and "Occlusion 40%" in text and "10 -> 20" in text
