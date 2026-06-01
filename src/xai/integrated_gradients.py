import torch
import numpy as np
from captum.attr import IntegratedGradients


def compute_integrated_gradients(model, image_tensor, target_class, n_steps=50):
    """
    Integrated Gradients attribution using Captum.

    Integrates gradients along a straight path from a black baseline to the input.
    More stable than vanilla saliency — satisfies completeness axiom.
    """
    model.eval()
    inp = image_tensor.unsqueeze(0)
    baseline = torch.zeros_like(inp)
    ig = IntegratedGradients(model)
    attr = ig.attribute(inp, baseline, target=int(target_class), n_steps=n_steps)
    attr = attr.squeeze().detach().cpu()
    if attr.dim() == 3:
        attr = attr.abs().mean(dim=0)
    return attr.numpy()
