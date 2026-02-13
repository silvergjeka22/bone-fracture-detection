"""
Training configuration settings.

This module contains hyperparameters and settings for model training.
"""

# Training Hyperparameters
TRAINING_CONFIG = {
    'batch_size': 32,
    'num_epochs': 30,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'momentum': 0.9,  # For SGD optimizer
}

# Cross-Validation Settings
CV_CONFIG = {
    'k_folds': 5,
    'shuffle': True,
    'random_state': 42,
}

# Optimizer Settings
OPTIMIZER_CONFIG = {
    'type': 'adam',  # Options: 'adam', 'sgd', 'adamw'
    'adam': {
        'lr': 0.001,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 1e-5,
    },
    'sgd': {
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 1e-5,
        'nesterov': True,
    },
    'adamw': {
        'lr': 0.001,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 0.01,
    }
}

# Learning Rate Scheduler Settings
SCHEDULER_CONFIG = {
    'type': 'reduce_on_plateau',  # Options: 'reduce_on_plateau', 'step', 'cosine'
    'reduce_on_plateau': {
        'mode': 'min',
        'factor': 0.5,
        'patience': 3,
        'min_lr': 1e-6,
    },
    'step': {
        'step_size': 10,
        'gamma': 0.1,
    },
    'cosine': {
        'T_max': 30,
        'eta_min': 1e-6,
    }
}

# Data Augmentation Settings
AUGMENTATION_CONFIG = {
    'train': {
        'random_horizontal_flip': True,
        'random_rotation': 10,  # degrees
        'random_crop': True,
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
        },
    },
    'val_test': {
        'resize': (224, 224),
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        }
    }
}

# Early Stopping Settings
EARLY_STOPPING_CONFIG = {
    'enabled': True,
    'patience': 10,
    'min_delta': 0.001,
}

# Device Settings
DEVICE_CONFIG = {
    'use_cuda': True,  # Use CUDA if available
    'use_mps': True,   # Use Apple Silicon MPS if available
    'device_ids': None,  # For multi-GPU training, e.g., [0, 1]
}

# Checkpoint Settings
CHECKPOINT_CONFIG = {
    'save_best_only': True,
    'save_frequency': 5,  # Save every N epochs
    'checkpoint_dir': 'outputs/models',
}

# Logging Settings
LOGGING_CONFIG = {
    'log_frequency': 10,  # Log every N batches
    'tensorboard': False,
    'wandb': False,
}


def get_training_config():
    """Get the complete training configuration."""
    return {
        'training': TRAINING_CONFIG.copy(),
        'cv': CV_CONFIG.copy(),
        'optimizer': OPTIMIZER_CONFIG.copy(),
        'scheduler': SCHEDULER_CONFIG.copy(),
        'augmentation': AUGMENTATION_CONFIG.copy(),
        'early_stopping': EARLY_STOPPING_CONFIG.copy(),
        'device': DEVICE_CONFIG.copy(),
        'checkpoint': CHECKPOINT_CONFIG.copy(),
        'logging': LOGGING_CONFIG.copy(),
    }
