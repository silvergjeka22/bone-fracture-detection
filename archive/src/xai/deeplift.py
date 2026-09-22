import torch
import numpy as np
from captum.attr import DeepLift


def compute_deeplift(model, image_tensor, target_class):
    """
    DeepLIFT attribution using Captum.

    Assigns attribution scores by comparing each neuron's activation
    to a reference (baseline) activation. Captures non-linear effects
    that vanilla gradients miss.

    Note: DeepLIFT may warn about BatchNorm layers; this is handled
    gracefully with a gradient×input fallback.
    """
    model.eval()
    inp = image_tensor.unsqueeze(0)
    baseline = torch.zeros_like(inp)
    try:
        dl = DeepLift(model)
        attr = dl.attribute(inp, baseline, target=int(target_class))
        attr = attr.squeeze().detach().cpu()
        if attr.dim() == 3:
            attr = attr.abs().mean(dim=0)
        return attr.numpy()
    except Exception as e:
        # Fallback: gradient × input (common when model has BatchNorm)
        print(f"  [DeepLIFT] Captum warning ({type(e).__name__}), using Gradient×Input fallback.")
        inp_g = inp.clone().detach().requires_grad_(True)
        output = model(inp_g)
        model.zero_grad()
        output[0, int(target_class)].backward()
        attr = (inp_g.grad * inp_g).squeeze().detach().cpu()
        if attr.dim() == 3:
            attr = attr.abs().mean(dim=0)
        return attr.numpy()
