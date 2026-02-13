"""
GradCAM (Gradient-weighted Class Activation Mapping) for XAI.

This module will implement GradCAM for visualizing which regions of an image
are important for the model's prediction.

TODO: Implement GradCAM
- Use Captum's LayerGradCam class
- Support automatic layer selection for different architectures
- Implement upsampling to match input image size

Reference:
    Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks 
    via Gradient-based Localization" (2017)
    
Example usage (to be implemented):
    from captum.attr import LayerGradCam
    
    # For CNN models
    layer_gc = LayerGradCam(model, model.conv4)
    attribution = layer_gc.attribute(input_tensor, target=target_class)
"""

import torch

# Uncomment when implementing:
# from captum.attr import LayerGradCam


def compute_gradcam(model, input_tensor, target_class, target_layer=None):
    """
    Compute GradCAM attribution.
    
    Args:
        model: PyTorch model
        input_tensor: Input image tensor
        target_class: Target class for attribution
        target_layer: Layer to compute GradCAM on (auto-detect if None)
        
    Returns:
        attribution: Attribution map (upsampled to input size)
        
    TODO: Implement using Captum's LayerGradCam
    """
    raise NotImplementedError(
        "GradCAM not yet implemented. "
        "Please implement using Captum library."
    )


def get_target_layer(model):
    """
    Automatically select the target layer for GradCAM.
    
    Args:
        model: PyTorch model
        
    Returns:
        target_layer: The last convolutional layer
        
    TODO: Implement layer detection for different architectures
    """
    raise NotImplementedError(
        "Automatic layer selection not yet implemented."
    )
