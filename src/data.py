"""Dataset, transforms and data loaders.

Expected folder layout (Kaggle "Bone Fracture Multi-Region X-ray Data"):

    DATA_DIR/
        train/{fractured, not fractured}/
        val/{fractured, not fractured}/      (not used: CV is done on train)
        test/{fractured, not fractured}/
        choosen_test/{fractured, not fractured}/   (the images explained with XAI)

Labels follow the sorted folder names: 0 = 'fractured', 1 = 'not fractured'.
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True  # a few X-rays in the dataset are truncated files

IMAGE_SIZE = 224
MEAN = [0.5, 0.5, 0.5]  # Normalize(MEAN, STD) maps pixels from [0, 1] to [-1, 1]
STD = [0.5, 0.5, 0.5]
IMG_EXTENSIONS = (".png", ".jpg", ".jpeg")


class BoneFractureDataset(Dataset):
    """Images stored as root_dir/<class_name>/<image file>."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_names = sorted(d.name for d in self.root_dir.iterdir() if d.is_dir())
        # sorted() makes the order (and therefore the CV folds) identical on every OS
        self.samples = [
            (path, label)
            for label, class_name in enumerate(self.class_names)
            for path in sorted((self.root_dir / class_name).iterdir())
            if path.name.endswith(IMG_EXTENSIONS)
        ]
        self.labels = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(train=False):
    """Resize + normalise; training images are also randomly rotated, flipped and jittered."""
    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        steps += [
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


def load_datasets(data_dir):
    """Train set twice (with / without augmentation, for the CV train / validation folds) and the test set."""
    data_dir = Path(data_dir)
    return {
        "train": BoneFractureDataset(data_dir / "train", get_transforms(train=True)),
        "train_eval": BoneFractureDataset(data_dir / "train", get_transforms(train=False)),
        "test": BoneFractureDataset(data_dir / "test", get_transforms(train=False)),
    }


def make_loader(dataset, batch_size=32, shuffle=False, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


def load_images(folder):
    """All images of a folder as one tensor (N, 3, H, W) plus their labels (N,)."""
    dataset = BoneFractureDataset(folder, get_transforms(train=False))
    images, labels = zip(*(dataset[i] for i in range(len(dataset))))
    return torch.stack(images), torch.tensor(labels)


def denormalize(image):
    """Normalised (3, H, W) tensor -> grey (H, W) numpy image in [0, 1], for plotting."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (image.detach().cpu() * std + mean).mean(0).clamp(0, 1).numpy()


def summarize(data_dir, splits=("train", "val", "test")):
    """Number of images per split and class."""
    rows = {}
    for split in splits:
        split_dir = Path(data_dir) / split
        if split_dir.is_dir():
            rows[split] = {c.name: sum(p.name.endswith(IMG_EXTENSIONS) for p in c.iterdir())
                           for c in sorted(split_dir.iterdir()) if c.is_dir()}
    table = pd.DataFrame(rows).T
    table["total"] = table.sum(axis=1)
    return table
