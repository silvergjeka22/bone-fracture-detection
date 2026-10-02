"""FracAtlas: load the X-rays (grey, 224x224) and their fracture boxes, find near-duplicates, split, build loaders.

Labels: 0 = fractured (the positive class), 1 = not fractured.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

ImageFile.LOAD_TRUNCATED_IMAGES = True  

IMAGE_SIZE = 224
CLASS_FOLDERS = ("Fractured", "Non_fractured")   # label 0, label 1
CLASS_NAMES = ["fractured", "not fractured"]
IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
MEAN, STD = 0.5, 0.5   
BLACK = -1.0           


def find_data_dir(root):
    """First folder under `root` (itself included) that has images/Fractured and images/Non_fractured."""
    root = Path(root)
    for dirpath, dirnames, _ in os.walk(root):
        if all((Path(dirpath) / "images" / folder).is_dir() for folder in CLASS_FOLDERS):
            return Path(dirpath)
        if len(Path(dirpath).relative_to(root).parts) >= 5:
            dirnames[:] = []  
    raise FileNotFoundError(
        f"No folder with images/Fractured and images/Non_fractured found under {root}. On Kaggle, attach the "
        "dataset 'mahmudulhasantasin/fracatlas-original-dataset'; locally, unzip FracAtlas into data/.")


@dataclass
class ImageSet:
    """Images in memory, with their labels and fracture boxes."""
    images: np.ndarray
    labels: np.ndarray
    paths: list          
    sizes: np.ndarray
    class_names: list
    boxes: list          

    def __len__(self):
        return len(self.labels)

    def subset(self, idx):
        idx = np.asarray(idx)
        return ImageSet(self.images[idx], self.labels[idx], [self.paths[i] for i in idx], self.sizes[idx],
                        self.class_names, [self.boxes[i] for i in idx])

    def tensors(self, idx=None):
        """Normalised float images (n, 1, S, S) and labels (n,), e.g. for the XAI methods."""
        idx = np.arange(len(self)) if idx is None else np.asarray(idx)
        x = torch.from_numpy(self.images[idx]).float().div(255).unsqueeze(1)
        return (x - MEAN) / STD, torch.from_numpy(self.labels[idx])


def _read(path, size):
    """Grey (size, size) uint8 image and original (width, height); (None, None) if the file is broken."""
    try:
        with Image.open(path) as image:
            width, height = image.size
            grey = image.convert("L").resize((size, size), Image.BILINEAR)
            return np.asarray(grey, dtype=np.uint8), (width, height)
    except Exception: 
        return None, None


def read_boxes(path, size):
    """Boxes of one YOLO file as a (k, 4) array (x0, y0, x1, y1) in pixels of the size x size image."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 4), np.float32)
    cx, cy, w, h = np.loadtxt(path, ndmin=2)[:, 1:5].T
    return (np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1).clip(0, 1) * size).astype(np.float32)


def load_dataset(data_dir, size=IMAGE_SIZE, cache_dir=None, max_per_class=None, seed=0, workers=8):
    """Every image and its fracture boxes as one ImageSet; decoded images are cached in cache_dir."""
    data_dir = Path(data_dir)
    cache = Path(cache_dir) / f"fracatlas_{size}px_{max_per_class or 'all'}.npz" if cache_dir else None
    if cache is not None and cache.exists():
        with np.load(cache, allow_pickle=False) as f:
            images, labels, paths, sizes = f["images"], f["labels"], list(f["paths"]), f["sizes"]
    else:
        rng = np.random.default_rng(seed)
        files = []
        for label, folder in enumerate(CLASS_FOLDERS):
            found = sorted(p for p in (data_dir / "images" / folder).iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
            if max_per_class is not None and len(found) > max_per_class:
                found = [found[i] for i in sorted(rng.choice(len(found), max_per_class, replace=False))]
            files += [(p, label) for p in found]
        with ThreadPoolExecutor(workers) as pool:  
            loaded = list(pool.map(lambda f: _read(f[0], size), files))
        keep = [i for i, (image, _) in enumerate(loaded) if image is not None]
        if len(keep) < len(files):
            print(f"skipped {len(files) - len(keep)} unreadable files")
        images = np.stack([loaded[i][0] for i in keep])
        labels = np.array([files[i][1] for i in keep], dtype=np.int64)
        paths = [str(files[i][0]) for i in keep]
        sizes = np.array([loaded[i][1] for i in keep], dtype=np.int64)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, images=images, labels=labels, paths=np.array(paths), sizes=sizes)

    yolo = next((d for d in sorted(data_dir.rglob("YOLO")) if d.is_dir()), None)
    if yolo is None:
        raise FileNotFoundError(f"No YOLO annotation folder under {data_dir}: the fracture boxes are missing.")
    boxes = [read_boxes(yolo / f"{Path(p).stem}.txt", size) for p in paths]
    return ImageSet(images, labels, paths, sizes, list(CLASS_NAMES), boxes)


def split_indices(labels, groups, val=0.15, test=0.15, seed=0):
    """Stratified train / val / test indices; near-duplicates (same group) never cross splits."""
    labels, groups = np.asarray(labels), np.asarray(groups)

    def carve(pool, fraction):
        splitter = StratifiedGroupKFold(n_splits=round(1 / fraction), shuffle=True, random_state=seed)
        rest, part = next(splitter.split(pool, labels[pool], groups[pool]))
        return pool[rest], pool[part]

    rest, test_idx = carve(np.arange(len(labels)), test)
    train_idx, val_idx = carve(rest, val / (1 - test))
    return {"train": np.sort(train_idx), "val": np.sort(val_idx), "test": np.sort(test_idx)}


GEOMETRY = v2.Compose([v2.RandomHorizontalFlip(), v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1))])
COLOR = v2.ColorJitter(brightness=0.2, contrast=0.2)
AUGMENT = v2.Compose([GEOMETRY, COLOR])


class XrayDataset(Dataset):
    """Images of an ImageSet (optionally only `indices`) as normalised (1, S, S) tensors."""

    def __init__(self, image_set, indices=None, augment=False):
        self.images, self.labels = image_set.images, image_set.labels
        self.indices = np.arange(len(image_set)) if indices is None else np.asarray(indices)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        x = torch.from_numpy(self.images[j]).unsqueeze(0).float().div(255)
        if self.augment:
            x = AUGMENT(x).clamp(0, 1)
        return (x - MEAN) / STD, int(self.labels[j])


def make_loader(image_set, indices=None, augment=False, batch_size=32, shuffle=False, num_workers=0):
    return DataLoader(XrayDataset(image_set, indices, augment), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)


def denormalize(x):
    """Normalised (1, H, W) or (H, W) tensor -> (H, W) numpy image in [0, 1], for plotting."""
    x = x.detach().cpu().float()
    return (x.squeeze(0) * STD + MEAN).clamp(0, 1).numpy()


def pick_images(labels, n_per_class, seed=0, allowed=None):
    """Indices of `n_per_class` random images of each class (class 0 first), among `allowed` ones."""
    labels = np.asarray(labels)
    allowed = np.ones(len(labels), bool) if allowed is None else np.asarray(allowed)
    rng = np.random.default_rng(seed)
    picked = []
    for c in np.unique(labels):
        preferred = np.flatnonzero((labels == c) & allowed)
        chosen = list(rng.choice(preferred, min(n_per_class, len(preferred)), replace=False))
        if len(chosen) < n_per_class:  # not enough allowed images: complete with the others of the class
            others = np.flatnonzero((labels == c) & ~allowed)
            chosen += list(rng.choice(others, min(n_per_class - len(chosen), len(others)), replace=False))
        picked += sorted(chosen)
    return np.array(picked, dtype=int)


def summary_table(sets):
    """Images per split and class, class balance, fracture boxes and original image size."""
    rows = {}
    for split, s in sets.items():
        counts = np.bincount(s.labels, minlength=len(s.class_names))
        rows[split] = {**{name: int(n) for name, n in zip(s.class_names, counts)},
                       "total": len(s),
                       f"% {s.class_names[0]}": round(100 * counts[0] / len(s), 1),
                       "fracture boxes": int(sum(len(b) for b in s.boxes)),
                       "median width (px)": int(np.median(s.sizes[:, 0])),
                       "median height (px)": int(np.median(s.sizes[:, 1])),
                       "mean intensity": round(float(s.images.mean()) / 255, 3)}
    return pd.DataFrame.from_dict(rows, orient="index")


def _thumbnails(images, size, device):
    """Zero-mean, unit-norm size x size thumbnails of the 8 flips/90° rotations: (8, N, size*size)."""
    small = torch.cat([F.adaptive_avg_pool2d(torch.from_numpy(images[i:i + 1024]).to(device).float()[:, None], size)[:, 0]
                       for i in range(0, len(images), 1024)])
    views = torch.stack([torch.rot90(v, k, dims=(1, 2)) for v in (small, small.flip(2)) for k in range(4)]).flatten(2)
    views = views - views.mean(-1, keepdim=True)
    return views / views.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def near_duplicates(images, threshold=0.97, size=32, device="cpu", chunk=512):
    """Pairs (i < j) of images showing the same X-ray (32x32 thumbnails, 8 flips/rotations, cosine >= threshold)."""
    views = _thumbnails(images, size, device)
    rows = []
    for start in range(0, views.shape[1], chunk):
        sim = torch.einsum("vnd,md->vnm", views[:, start:start + chunk], views[0]).amax(0)  
        i_global = torch.arange(start, start + sim.shape[0], device=sim.device)[:, None]
        sim = sim.masked_fill(torch.arange(sim.shape[1], device=sim.device)[None] <= i_global, -1)  
        i, j = torch.nonzero(sim >= threshold, as_tuple=True)
        rows.append(np.column_stack([(i + start).cpu().numpy(), j.cpu().numpy(), sim[i, j].cpu().numpy()]))
    pairs = np.concatenate(rows) if rows else np.zeros((0, 3))
    return pd.DataFrame({"i": pairs[:, 0].astype(int), "j": pairs[:, 1].astype(int), "similarity": pairs[:, 2]})


def duplicate_groups(n, pairs):
    """Group id of each of n images: near-duplicates (directly or through a chain) share a group."""
    graph = coo_matrix((np.ones(len(pairs)), (pairs["i"], pairs["j"])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def study_duplicates(image_set, threshold=0.97, device="cpu"):
    """Near-duplicates in the dataset: a group id per image (copies share one), the pairs, a summary table."""
    pairs = near_duplicates(image_set.images, threshold=threshold, device=device)
    groups = duplicate_groups(len(image_set), pairs)
    sizes = np.bincount(groups)
    table = pd.DataFrame({"images": [len(image_set)],
                          "with a near-duplicate": [int((sizes[groups] > 1).sum())],
                          "duplicate groups": [int((sizes > 1).sum())],
                          "largest group": [int(sizes.max())]}, index=["dataset"])
    table["%"] = (100 * table["with a near-duplicate"] / table["images"]).round(1)
    return {"groups": groups, "pairs": pairs, "table": table}
