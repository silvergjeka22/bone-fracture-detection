"""
Custom CNN model for bone fracture detection.

This module contains the custom-designed 2D Convolutional Neural Network
for binary classification of bone fracture images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BoneFractureCNN(nn.Module):
    """
    2D Convolutional Neural Network for Bone Fracture Classification
    Architecture: 4 conv layers with increasing filters [32, 64, 128, 256]
    """

    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(BoneFractureCNN, self).__init__()

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 4
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Calculate flattened size after convolutions
        # Input: 224x224 -> after 4 pooling layers: 224/16 = 14x14
        self.flatten_size = 256 * 14 * 14

        # Fully Connected Layers (Classifier)
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Conv Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x

    def get_conv1_filters(self):
        """Extract first convolutional layer filters for visualization"""
        return self.conv1.weight.data.clone()
