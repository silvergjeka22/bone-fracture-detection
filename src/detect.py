"""YOLOv8 (Ultralytics) trained on the radiologists' fracture boxes: the detector of the pipeline.

ultralytics is imported inside the functions, so the rest of the project does not need it.
"""

import shutil
from pathlib import Path

import numpy as np

CLASS_NAME = "fracture"


def write_yolo_dataset(sets, out_dir):
    """Copy the fractured X-rays of every split with their boxes in YOLO format; returns data.yaml."""
    out_dir = Path(out_dir).resolve()
    shutil.rmtree(out_dir, ignore_errors=True)
    for split, image_set in sets.items():
        (out_dir / "images" / split).mkdir(parents=True)
        (out_dir / "labels" / split).mkdir(parents=True)
        size = image_set.images.shape[1]  # the boxes are in pixels of the resized image
        for path, boxes in zip(image_set.paths, image_set.boxes):
            if len(boxes) == 0:
                continue
            path = Path(path)
            shutil.copy(path, out_dir / "images" / split / path.name)
            x0, y0, x1, y1 = (np.asarray(boxes, dtype=float) / size).T
            lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                     for cx, cy, w, h in zip((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)]
            (out_dir / "labels" / split / f"{path.stem}.txt").write_text("\n".join(lines) + "\n")
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(f"path: {out_dir}\n" + "".join(f"{split}: images/{split}\n" for split in sets)
                         + f"names:\n  0: {CLASS_NAME}\n")
    return data_yaml


def train(data_yaml, out_dir, model="yolov8s.pt", epochs=50, imgsz=640, batch=16, seed=0, workers=2):
    """Train YOLO (COCO-pretrained weights, or a .yaml to start from scratch); returns <out_dir>/best.pt."""
    from ultralytics import YOLO

    out_dir = Path(out_dir).resolve()
    detector = YOLO(model)
    detector.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, seed=seed, workers=workers,
                   project=str(out_dir), name="yolo", exist_ok=True, plots=False, verbose=False)
    best = Path(getattr(detector.trainer, "best", out_dir / "yolo" / "weights" / "best.pt"))
    shutil.copy(best, out_dir / "best.pt")
    return out_dir / "best.pt"


def load(out_dir):
    """Weights saved by `train` (TRAIN = False)."""
    path = Path(out_dir) / "best.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the notebook with TRAIN = True first.")
    return path


def box_metrics(weights, data_yaml, imgsz=640, split="test"):
    """Box metrics of the detector on the fractured X-rays of `split`."""
    from ultralytics import YOLO

    metrics = YOLO(str(weights)).val(data=str(data_yaml), split=split, imgsz=imgsz, project=str(Path(weights).parent),
                                     name=f"val_{split}", exist_ok=True, plots=False, verbose=False)
    box = metrics.box
    return {"mAP@0.5": float(box.map50), "mAP@0.5:0.95": float(box.map),
            "precision": float(box.mp), "recall": float(box.mr)}


def to_boxes(xyxyn, confidence, size):
    """YOLO's normalised boxes + confidences -> (k, 5) array in pixels of our size x size image, most confident first."""
    xyxyn, confidence = np.asarray(xyxyn, dtype=float).reshape(-1, 4), np.asarray(confidence, dtype=float)
    order = np.argsort(-confidence, kind="stable")
    return np.column_stack([xyxyn[order] * size, confidence[order]]).astype(np.float32)


def find_boxes(weights, paths, size, conf=0.25, imgsz=640):
    """Boxes found in each image (original files in `paths`): list of (k, 5) arrays, see `to_boxes`."""
    from ultralytics import YOLO

    results = YOLO(str(weights)).predict(source=[str(p) for p in paths], conf=conf, imgsz=imgsz, stream=True,
                                         verbose=False)
    return [to_boxes(r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy(), size) for r in results]
