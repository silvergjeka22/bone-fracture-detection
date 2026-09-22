import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import random

class BoneFractureDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []

        # Get class names (folders in root_dir)
        self.class_names = sorted([d for d in os.listdir(root_dir)
                                   if os.path.isdir(os.path.join(root_dir, d))])

        # Create class to index mapping
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}

        # Load all image paths and labels
        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)

        # Print dataset statistics
        print(f"Loaded {len(self.images)} images from {root_dir}")
        print(f"Classes: {self.class_names}")
        print(f"Class distribution: {dict(zip(self.class_names, [self.labels.count(i) for i in range(len(self.class_names))]))}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self):
        class_counts = [self.labels.count(i) for i in range(len(self.class_names))]
        total_samples = len(self.labels)
        weights = [total_samples / (len(self.class_names) * count) for count in class_counts]
        return weights

    def get_sample_by_class(self, class_name, num_samples=1):
        class_idx = self.class_to_idx[class_name]
        class_indices = [i for i, label in enumerate(self.labels) if label == class_idx]
        sampled_indices = random.sample(class_indices, min(num_samples, len(class_indices)))
        return [(self.images[i], self.labels[i]) for i in sampled_indices]
