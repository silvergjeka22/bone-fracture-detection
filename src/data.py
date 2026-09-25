"""Data: find the dataset, load and cache the images, study the dataset, build the data loaders.

Kaggle dataset "Bone Fracture Multi-Region X-ray Data" (bmadushanirodrigo/fracture-multi-region-x-ray-data):

    <data_dir>/train/{fractured, not fractured}/   ~9.2k images: cross-validation + final training
    <data_dir>/val/{fractured, not fractured}/      ~0.8k images: monitoring of the final training
    <data_dir>/test/{fractured, not fractured}/     ~0.5k images: final evaluation only

X-rays are grey, so every image is loaded as ONE channel, resized to 224x224 and kept in memory
as uint8 (about 0.5 GB for the whole dataset), which makes every epoch GPU-bound.
Labels follow the sorted folder names: 0 = 'fractured', 1 = 'not fractured'.
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
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

ImageFile.LOAD_TRUNCATED_IMAGES = True  # a few X-rays in the dataset are truncated files

IMAGE_SIZE = 224
SPLITS = ("train", "val", "test")
IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
MEAN, STD = 0.5, 0.5   # pixels [0, 1] -> [-1, 1]
BLACK = -1.0           # a black pixel after normalisation: the "removed" value used by the XAI methods


# ----------------------------------------------------------------------------- loading

def find_data_dir(root):
    """First folder under `root` (itself included) that has `train/` and `test/` subfolders."""
    root = Path(root)
    for dirpath, dirnames, _ in os.walk(root):
        if {"train", "test"} <= set(dirnames):
            return Path(dirpath)
        if len(Path(dirpath).relative_to(root).parts) >= 5:
            dirnames[:] = []  # do not walk deeper
    raise FileNotFoundError(
        f"No folder with train/ and test/ found under {root}. On Kaggle, attach the dataset "
        "'bmadushanirodrigo/fracture-multi-region-x-ray-data'; locally, download it into data/.")


@dataclass
class ImageSet:
    """All images of one split in memory."""
    images: np.ndarray   # (N, S, S) uint8, grey
    labels: np.ndarray   # (N,) int64
    paths: list          # file path of every image
    sizes: np.ndarray    # (N, 2) original width and height in pixels
    class_names: list

    def __len__(self):
        return len(self.labels)

    def subset(self, idx):
        idx = np.asarray(idx)
        return ImageSet(self.images[idx], self.labels[idx], [self.paths[i] for i in idx],
                        self.sizes[idx], self.class_names)

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
    except Exception:  # corrupted file: skip it (and report it)
        return None, None


def load_split(split_dir, size=IMAGE_SIZE, cache_dir=None, max_per_class=None, seed=0, workers=8):
    """Load one split (split_dir/<class>/<image>) into an ImageSet, cached as .npz in `cache_dir`.

    `max_per_class` keeps a random subset of each class (quick test runs).
    """
    split_dir = Path(split_dir)
    class_names = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"{split_dir.name}_{size}px_{max_per_class or 'all'}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as f:
                return ImageSet(f["images"], f["labels"], list(f["paths"]), f["sizes"], list(f["class_names"]))

    rng = np.random.default_rng(seed)
    files = []
    for label, name in enumerate(class_names):
        paths = sorted(p for p in (split_dir / name).iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
        if max_per_class is not None and len(paths) > max_per_class:
            paths = [paths[i] for i in sorted(rng.choice(len(paths), max_per_class, replace=False))]
        files += [(p, label) for p in paths]

    with ThreadPoolExecutor(workers) as pool:  # PIL releases the GIL while decoding
        loaded = list(pool.map(lambda f: _read(f[0], size), files))
    keep = [i for i, (image, _) in enumerate(loaded) if image is not None]
    if len(keep) < len(files):
        print(f"{split_dir.name}: skipped {len(files) - len(keep)} unreadable files")

    image_set = ImageSet(
        images=np.stack([loaded[i][0] for i in keep]),
        labels=np.array([files[i][1] for i in keep], dtype=np.int64),
        paths=[str(files[i][0]) for i in keep],
        sizes=np.array([loaded[i][1] for i in keep], dtype=np.int64),
        class_names=class_names,
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, images=image_set.images, labels=image_set.labels, paths=np.array(image_set.paths),
                 sizes=image_set.sizes, class_names=np.array(class_names))
    return image_set


def load_splits(data_dir, splits=SPLITS, **kwargs):
    """{split: ImageSet} for every split folder that exists in data_dir."""
    sets = {s: load_split(Path(data_dir) / s, **kwargs) for s in splits if (Path(data_dir) / s).is_dir()}
    names = {tuple(s.class_names) for s in sets.values()}
    if len(names) != 1:
        raise ValueError(f"The splits have different class folders: {names}")
    return sets


# ----------------------------------------------------------------------------- training data

# Random, anatomically plausible changes, applied only to training images: left/right flip,
# small rotation / shift / zoom (the empty border is black, like the X-ray background) and
# brightness / contrast (different X-ray machines and exposures).
AUGMENT = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
])


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


# ----------------------------------------------------------------------------- dataset study

def summary_table(sets):
    """Images per split and class, class balance and original image size."""
    rows = {}
    for split, s in sets.items():
        counts = np.bincount(s.labels, minlength=len(s.class_names))
        rows[split] = {**{name: int(n) for name, n in zip(s.class_names, counts)},
                       "total": len(s),
                       f"% {s.class_names[0]}": round(100 * counts[0] / len(s), 1),
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


def near_duplicates(images_a, images_b=None, threshold=0.97, size=32, device="cpu", chunk=512):
    """Pairs of images that show the same X-ray: DataFrame(i, j, similarity).

    Two images match when the cosine similarity of their 32x32 zero-mean thumbnails is >= threshold
    for one of the 8 flips/90° rotations. This finds copies that were resized, re-compressed,
    flipped, rotated by 90° or changed in brightness/contrast; copies rotated by other angles or
    cropped are missed, so the counts are a lower bound. With images_b=None: pairs inside images_a (i < j).
    """
    a = _thumbnails(images_a, size, device)
    b = a[0] if images_b is None else _thumbnails(images_b, size, device)[0]
    rows = []
    for start in range(0, a.shape[1], chunk):
        sim = torch.einsum("vnd,md->vnm", a[:, start:start + chunk], b).amax(0)  # best of the 8 views
        if images_b is None:  # every pair once, not with itself
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


def study_duplicates(sets, threshold=0.97, device="cpu"):
    """Near-duplicates inside the training set and between the other splits and the training set.

    Returns groups (a group id per training image, for group-aware cross-validation), the pairs,
    a boolean `leaked[split]` (image has a near-duplicate in train) and a summary table.
    """
    train = sets["train"]
    pairs = {"train": near_duplicates(train.images, threshold=threshold, device=device)}
    groups = duplicate_groups(len(train), pairs["train"])
    sizes = np.bincount(groups)
    rows = {"train": {"images": len(train),
                      "with a near-duplicate in train": int((sizes[groups] > 1).sum()),
                      "duplicate groups": int((sizes > 1).sum()),
                      "largest group": int(sizes.max())}}
    leaked = {}
    for split in sets:
        if split == "train":
            continue
        pairs[split] = near_duplicates(sets[split].images, train.images, threshold, device=device)
        leaked[split] = np.isin(np.arange(len(sets[split])), pairs[split]["i"])
        rows[split] = {"images": len(sets[split]), "with a near-duplicate in train": int(leaked[split].sum())}
    table = pd.DataFrame.from_dict(rows, orient="index").astype("Int64")
    table["%"] = (100 * table["with a near-duplicate in train"] / table["images"]).round(1)
    return {"groups": groups, "pairs": pairs, "leaked": leaked, "table": table}
