"""A tiny synthetic stand-in for FracAtlas, used by the tests and for dry runs.

Images look vaguely like X-rays: dark background, a bright "bone" bar at a random angle, a white
marker in a corner; in the 'fractured' class the bone is crossed by a dark crack, and the box of the
crack is saved in YOLO format, like the radiologists' boxes. One healthy X-ray is saved again flipped
and rotated, one fractured X-ray brighter (near-duplicates), so the duplicate study has something to find.

    root/images/Fractured/IMG*.jpg    root/images/Non_fractured/IMG*.jpg    root/Annotations/YOLO/IMG*.txt

    python -m tests.synthetic <folder> [--per-class 40]
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

FOLDERS = ("Fractured", "Non_fractured")


def xray(fractured, rng, size=None):
    """(image, box): box = (cx, cy, w, h) of the crack as fractions of the image, None if not fractured."""
    size = size or int(rng.integers(180, 420))
    image = Image.new("L", (size, size), int(rng.integers(0, 25)))
    draw = ImageDraw.Draw(image)
    angle = rng.uniform(0, np.pi)
    c = np.array([size / 2, size / 2]) + rng.uniform(-0.1, 0.1, 2) * size
    d = np.array([np.cos(angle), np.sin(angle)])
    n = np.array([-d[1], d[0]])
    length, width = size * rng.uniform(0.3, 0.45), size * rng.uniform(0.06, 0.1)
    corners = [c + s1 * length * d + s2 * width * n for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    draw.polygon([tuple(p) for p in corners], fill=int(rng.integers(170, 230)))
    box = None
    if fractured:  # a dark crack across the bone
        t = rng.uniform(-0.5, 0.5) * length
        tilt = rng.uniform(-0.4, 0.4)
        a, b = c + t * d - 1.4 * width * (n + tilt * d), c + t * d + 1.4 * width * (n - tilt * d)
        line = max(2, size // 60)
        draw.line([tuple(a), tuple(b)], fill=int(rng.integers(0, 40)), width=line)
        (x0, y0), (x1, y1) = (np.minimum(a, b) - line) / size, (np.maximum(a, b) + line) / size
        x0, y0, x1, y1 = np.clip([x0, y0, x1, y1], 0, 1)
        box = ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)
    corner = rng.integers(0, 4)
    x0 = size * 0.05 if corner % 2 == 0 else size * 0.82
    y0 = size * 0.05 if corner < 2 else size * 0.82
    draw.rectangle([x0, y0, x0 + size * 0.1, y0 + size * 0.1], fill=240)
    noise = rng.normal(0, 6, (size, size))
    return Image.fromarray(np.clip(np.asarray(image, float) + noise, 0, 255).astype(np.uint8)), box


def make_dataset(root, per_class=16, seed=0):
    """FracAtlas-like folder with `per_class` X-rays per class (+ 3 planted copies); returns root."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    yolo = root / "Annotations" / "YOLO"
    yolo.mkdir(parents=True, exist_ok=True)
    k = 0
    for label, folder in enumerate(FOLDERS):
        (root / "images" / folder).mkdir(parents=True, exist_ok=True)
        for _ in range(per_class):
            k += 1
            image, box = xray(label == 0, rng)
            image.save(root / "images" / folder / f"IMG{k:07d}.jpg", quality=95)
            if box is not None:
                (yolo / f"IMG{k:07d}.txt").write_text("0 " + " ".join(f"{v:.6f}" for v in box) + "\n")
    # near-duplicates: a healthy X-ray flipped and rotated, a fractured one brighter (same fracture box)
    healthy = root / "images" / "Non_fractured" / f"IMG{per_class + 1:07d}.jpg"
    with Image.open(healthy) as source:
        source.transpose(Image.FLIP_LEFT_RIGHT).save(healthy.with_name(f"{healthy.stem}_flip.jpg"), quality=95)
        source.transpose(Image.ROTATE_90).resize((300, 300)).save(healthy.with_name(f"{healthy.stem}_rot.jpg"),
                                                                  quality=95)
    fractured = root / "images" / "Fractured" / "IMG0000001.jpg"
    with Image.open(fractured) as source:
        ImageEnhance.Brightness(source).enhance(1.15).save(fractured.with_name("IMG0000001_bright.jpg"), quality=90)
    shutil.copy(yolo / "IMG0000001.txt", yolo / "IMG0000001_bright.txt")
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--per-class", type=int, default=40)
    args = parser.parse_args()
    print(make_dataset(args.root, args.per_class))
