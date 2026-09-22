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


def input_x_gradient_map(model, image_tensor, target_class=None, device='cuda'):

    model.eval()
    input_image = image_tensor.unsqueeze(0).to(device)
    input_image.requires_grad_(True)

    if target_class is None:
        with torch.no_grad():
            pred = model(input_image)
            target_class = pred.argmax(dim=1).item()

    input_x_gradient = InputXGradient(model)

    attributions = input_x_gradient.attribute(
        input_image,
        target=target_class,
    )

    attributions = attributions.squeeze()

    attributions = attributions.mean(0)

    return attributions.detach().cpu().numpy()