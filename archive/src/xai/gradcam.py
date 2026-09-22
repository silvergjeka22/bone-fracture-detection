import torch
import torch.nn.functional as F
import numpy as np
from captum.attr import LayerGradCam


def compute_gradcam(model, image_tensor, target_class, target_layer=None):
    """
    GradCAM attribution using Captum LayerGradCam.

    Highlights class-discriminative spatial regions by weighting
    feature maps of a convolutional layer by their gradient.

    Args:
        model: trained CNN or ScatNet
        image_tensor: (C, H, W) input tensor (no batch dim)
        target_class: int, predicted or true class index
        target_layer: nn.Module — e.g. model.conv4 for CNN.
                      Pass None for ScatNet (GradCAM NOT applicable).

    Returns:
        (H, W) numpy array, or None if target_layer is None.

    Note:
        GradCAM CANNOT be applied to ScatNet because ScatNet has no
        learnable convolutional feature maps — its feature extraction
        uses fixed Morlet scattering wavelets. No gradient flows through
        a spatial conv layer, so LayerGradCam has no layer to hook into.
    """
    if target_layer is None:
        return None

    model.eval()
    inp = image_tensor.unsqueeze(0)
    lgc = LayerGradCam(model, target_layer)
    attr = lgc.attribute(inp, target=int(target_class))
    # Upsample to original input resolution
    attr = F.interpolate(attr, size=(224, 224), mode='bilinear', align_corners=False)
    attr = attr.squeeze().detach().cpu()
    if attr.dim() == 3:
        attr = attr.mean(dim=0)
    return torch.clamp(attr, min=0).numpy()
