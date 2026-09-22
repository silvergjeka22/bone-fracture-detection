import torch
import numpy as np
from captum.attr import IntegratedGradients

def compute_integrated_gradients(model, image_tensor, target_class=None, device='cuda'):
    """
    Integrated Gradients attribution using Captum.

    Integrates gradients along a straight path from a black baseline to the input.
    More stable than vanilla saliency — satisfies completeness axiom.
    """

    model.eval()

    input_image = image_tensor.unsqueeze(0).to(device)

    if target_class is None:
        with torch.no_grad():
            target_class = model(input_image).argmax(dim=1).item()

    ig = IntegratedGradients(model)

    baseline = torch.zeros_like(input_image)

    attributions = ig.attribute(
        input_image,
        baselines=baseline,
        target=target_class,
        n_steps=100
    )

    attributions = attributions.squeeze(0)      # (C,H,W)
    attributions = attributions.mean(0)   # (H,W)

    return attributions.cpu().numpy()