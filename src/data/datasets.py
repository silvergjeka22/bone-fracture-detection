import os
from PIL import Image
from torch.utils.data import Dataset

class BoneFractureDataset(Dataset):
    """
    Custom Dataset for Bone Fracture Classification.

    This dataset class loads images from a directory structure where each
    subdirectory represents a class (e.g., 'fractured', 'not_fractured').

    Directory structure expected:
        root_dir/
        ├── class1/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        └── class2/
            ├── image1.jpg
            └── ...

    Args:
        root_dir (str): Root directory containing class subdirectories
        transform (callable, optional): Optional transform to be applied to images

    Attributes:
        root_dir (str): Root directory path
        transform (callable): Image transformations
        images (list): List of all image paths
        labels (list): List of corresponding labels
        class_names (list): List of class names
        class_to_idx (dict): Mapping from class names to indices
    """

    def __init__(self, root_dir, transform=None):
        """
        Initialize the BoneFractureDataset.

        Args:
            root_dir: Root directory containing class subdirectories
            transform: Optional transform to be applied to images
        """
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
        """
        Return the total number of images in the dataset.

        Returns:
            int: Number of images
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get an image and its label by index.

        Args:
            idx (int): Index of the image to retrieve

        Returns:
            tuple: (image, label) where image is a transformed PIL Image or Tensor
                   and label is an integer class index
        """
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self):
        """
        Calculate class weights for handling imbalanced datasets.
        Useful for weighted loss functions.

        Returns:
            list: Weight for each class (inversely proportional to frequency)
        """
        import numpy as np
        class_counts = [self.labels.count(i) for i in range(len(self.class_names))]
        total_samples = len(self.labels)
        weights = [total_samples / (len(self.class_names) * count) for count in class_counts]
        return weights

    def get_sample_by_class(self, class_name, num_samples=1):
        """
        Get random samples from a specific class.

        Args:
            class_name (str): Name of the class
            num_samples (int): Number of samples to retrieve

        Returns:
            list: List of (image_path, label) tuples
        """
        import random
        class_idx = self.class_to_idx[class_name]
        class_indices = [i for i, label in enumerate(self.labels) if label == class_idx]
        sampled_indices = random.sample(class_indices, min(num_samples, len(class_indices)))
        return [(self.images[i], self.labels[i]) for i in sampled_indices]
