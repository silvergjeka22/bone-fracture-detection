"""A tiny synthetic stand-in for the X-ray dataset, used by the tests and for dry runs.

Images look vaguely like X-rays: dark background, a bright "bone" bar at a random angle, a
white marker in a corner; in the 'fractured' class the bone is crossed by a dark crack.
A few training images are saved again flipped / rotated / brightened (near-duplicates), and a
few test images are copies of training images, so the duplicate study has something to find.

    python -m tests.synthetic <folder> [--per-class 40]
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

CLASSES = ("fractured", "not fractured")


def xray(fractured, rng, size=None):
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
    if fractured:  # a dark crack across the bone
        t = rng.uniform(-0.5, 0.5) * length
        tilt = rng.uniform(-0.4, 0.4)
        a, b = c + t * d - 1.4 * width * (n + tilt * d), c + t * d + 1.4 * width * (n - tilt * d)
        draw.line([tuple(a), tuple(b)], fill=int(rng.integers(0, 40)), width=max(2, size // 60))
    corner = rng.integers(0, 4)
    x0 = size * 0.05 if corner % 2 == 0 else size * 0.82
    y0 = size * 0.05 if corner < 2 else size * 0.82
    draw.rectangle([x0, y0, x0 + size * 0.1, y0 + size * 0.1], fill=240)
    noise = rng.normal(0, 6, (size, size))
    return Image.fromarray(np.clip(np.asarray(image, float) + noise, 0, 255).astype(np.uint8))


def make_dataset(root, per_class=(40, 10, 10), seed=0):
    """root/{train,val,test}/{fractured,not fractured}/*.png; returns root."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    for split, n in zip(("train", "val", "test"), per_class):
        for label, name in enumerate(CLASSES):
            folder = root / split / name
            folder.mkdir(parents=True, exist_ok=True)
            for k in range(n):
                xray(label == 0, rng).save(folder / f"{split}_{label}_{k:03d}.png")
    # near-duplicates: copies of the first training images, flipped / rotated / brighter
    for label, name in enumerate(CLASSES):
        source = Image.open(root / "train" / name / f"train_{label}_000.png")
        source.transpose(Image.FLIP_LEFT_RIGHT).save(root / "train" / name / f"train_{label}_dup_flip.png")
        source.transpose(Image.ROTATE_90).resize((300, 300)).save(root / "train" / name / f"train_{label}_dup_rot.png")
        ImageEnhance.Brightness(source).enhance(1.15).save(root / "test" / name / f"test_{label}_leak.jpg", quality=90)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--per-class", type=int, default=40)
    args = parser.parse_args()
    print(make_dataset(args.root, (args.per_class, max(args.per_class // 4, 4), max(args.per_class // 4, 4))))
