import torch
import numpy as np
from captum.attr import Saliency


def compute_saliency_captum(model, image_tensor, target_class):
    """
    Vanilla Gradient / Saliency map using Captum.
    Highlights pixels that most influence the output when perturbed.
    """
    model.eval()
    inp = image_tensor.unsqueeze(0).requires_grad_(True)
    sal = Saliency(model)
    attr = sal.attribute(inp, target=int(target_class))
    attr = attr.squeeze().detach().cpu()
    if attr.dim() == 3:
        attr = attr.abs().mean(dim=0)
    return attr.numpy()


def compute_saliency_manual(model, image_tensor, target_class):
    """
    Vanilla Gradient / Saliency implemented manually (pure PyTorch).
    Gradient of the target class score w.r.t. the input image.
    """
    model.eval()
    inp = image_tensor.unsqueeze(0).clone().detach().requires_grad_(True)
    output = model(inp)
    model.zero_grad()
    output[0, int(target_class)].backward()
    attr = inp.grad.data.abs().squeeze()
    if attr.dim() == 3:
        attr = attr.mean(dim=0)
    return attr.cpu().numpy()
