"""
Models module - Neural network architectures for bone fracture detection.

Available models:
- BoneFractureCNN: Custom CNN architecture
- BoneFractureResNet18: ResNet18 with custom classifier
- BoneFractureResNet50: ResNet50 with custom classifier
- BoneFractureVGG16: VGG16 with custom classifier
- BoneFractureVGG19: VGG19 with custom classifier
- BoneFractureScatNet: ScatNet architecture (to be implemented)
"""

from .base import BaseModel
from .cnn import BoneFractureCNN
from .pretrained import (
    BoneFractureResNet18,
    BoneFractureResNet50,
    BoneFractureVGG16,
    BoneFractureVGG19
)

__all__ = [
    'BaseModel',
    'BoneFractureCNN',
    'BoneFractureResNet18',
    'BoneFractureResNet50',
    'BoneFractureVGG16',
    'BoneFractureVGG19',
]
