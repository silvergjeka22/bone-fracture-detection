"""
Model configuration settings.

This module contains configuration parameters for different model architectures.
"""

# CNN Model Configuration
CNN_CONFIG = {
    'num_classes': 2,
    'dropout_rate': 0.5,
    'input_size': (224, 224),
    'channels': [32, 64, 128, 256],
}

# ResNet Configuration
RESNET18_CONFIG = {
    'num_classes': 2,
    'pretrained': True,
    'freeze_backbone': False,  # Set to True to freeze pre-trained weights
}

RESNET50_CONFIG = {
    'num_classes': 2,
    'pretrained': True,
    'freeze_backbone': False,
}

# VGG Configuration
VGG16_CONFIG = {
    'num_classes': 2,
    'pretrained': True,
    'freeze_features': False,  # Set to True to freeze feature extractor
}

VGG19_CONFIG = {
    'num_classes': 2,
    'pretrained': True,
    'freeze_features': False,
}

# ScatNet Configuration
SCATNET_CONFIG = {
    'num_classes': 2,
    'J': 2,  # Number of scales
    'L': 8,  # Number of angles
    'input_size': (224, 224),
}

# Model Registry - Maps model names to their configurations
MODEL_CONFIGS = {
    'cnn': CNN_CONFIG,
    'resnet18': RESNET18_CONFIG,
    'resnet50': RESNET50_CONFIG,
    'vgg16': VGG16_CONFIG,
    'vgg19': VGG19_CONFIG,
    'scatnet': SCATNET_CONFIG,
}


def get_model_config(model_name):
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Name of the model (e.g., 'cnn', 'resnet18')
        
    Returns:
        Dictionary containing model configuration
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Available models: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name].copy()
