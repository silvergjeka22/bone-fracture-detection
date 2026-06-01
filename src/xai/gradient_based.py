import torch
import numpy as np
from captum.attr import InputXGradient


def compute_gradient_x_input(model, image_tensor, target_class):
    """
    Gradient × Input attribution using Captum.

    Multiplies the input tensor element-wise by the gradient of the
    target class score. Combines gradient sensitivity with input magnitude,
    filtering out regions with low input values even if gradient is large.
    """
    model.eval()
    inp = image_tensor.unsqueeze(0)
    gxi = InputXGradient(model)
    attr = gxi.attribute(inp, target=int(target_class))
    attr = attr.squeeze().detach().cpu()
    if attr.dim() == 3:
        attr = attr.abs().mean(dim=0)
    return attr.numpy()
