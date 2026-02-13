"""
Pre-trained models for bone fracture detection.

This module contains transfer learning models based on popular architectures
(ResNet, VGG) pre-trained on ImageNet and fine-tuned for bone fracture classification.
"""

import torch
import torch.nn as nn
from torchvision import models
from .base import BaseModel


class BoneFractureResNet18(BaseModel):
    """
    ResNet18 for Bone Fracture Classification
    Uses pre-trained weights from ImageNet
    """

    def __init__(self, num_classes=2, pretrained=True):
        super(BoneFractureResNet18, self).__init__()

        # Load pre-trained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)

        # Get number of features from last layer
        num_features = self.resnet.fc.in_features

        # Replace final fully connected layer
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

    def get_conv1_filters(self):
        """Extract first convolutional layer filters"""
        return self.resnet.conv1.weight.data.clone()


class BoneFractureResNet50(BaseModel):
    """
    ResNet50 for Bone Fracture Classification
    Uses pre-trained weights from ImageNet
    """

    def __init__(self, num_classes=2, pretrained=True):
        super(BoneFractureResNet50, self).__init__()

        # Load pre-trained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)

        # Get number of features from last layer
        num_features = self.resnet.fc.in_features

        # Replace final fully connected layer
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

    def get_conv1_filters(self):
        """Extract first convolutional layer filters"""
        return self.resnet.conv1.weight.data.clone()


class BoneFractureVGG16(BaseModel):
    """
    VGG16 for Bone Fracture Classification
    Uses pre-trained weights from ImageNet
    """

    def __init__(self, num_classes=2, pretrained=True):
        super(BoneFractureVGG16, self).__init__()

        # Load pre-trained VGG16
        self.vgg = models.vgg16(pretrained=pretrained)

        # Get number of features from classifier
        num_features = self.vgg.classifier[6].in_features

        # Replace final classifier layer
        self.vgg.classifier[6] = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.vgg(x)

    def get_conv1_filters(self):
        """Extract first convolutional layer filters"""
        return self.vgg.features[0].weight.data.clone()


class BoneFractureVGG19(BaseModel):
    """
    VGG19 for Bone Fracture Classification
    Uses pre-trained weights from ImageNet
    """

    def __init__(self, num_classes=2, pretrained=True):
        super(BoneFractureVGG19, self).__init__()

        # Load pre-trained VGG19
        self.vgg = models.vgg19(pretrained=pretrained)

        # Get number of features from classifier
        num_features = self.vgg.classifier[6].in_features

        # Replace final classifier layer
        self.vgg.classifier[6] = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.vgg(x)

    def get_conv1_filters(self):
        """Extract first convolutional layer filters"""
        return self.vgg.features[0].weight.data.clone()
