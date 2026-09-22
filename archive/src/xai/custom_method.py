"""
Guided Backpropagation — implemented from scratch using PyTorch hooks.

This is the custom XAI method required by the exam. It is compared against
Captum's GuidedBackprop to validate correctness.

Reference: Springenberg et al. (2014) "Striving for Simplicity:
           The All Convolutional Net", ICLR Workshop.

Key idea:
    During standard backpropagation, negative gradients through ReLU are
    zeroed out (standard ReLU rule). Guided Backprop additionally zeros out
    gradients that are positive in the backward signal but correspond to
    negative forward activations. This produces sharper, higher-resolution
    attribution maps than vanilla saliency.

    Guided backprop rule for ReLU:
        grad_in = grad_out  if grad_out > 0 AND forward_act > 0
        grad_in = 0         otherwise

    Implemented by registering a backward hook on every ReLU layer that
    clamps the gradient to be non-negative (the forward-activation condition
    is already handled by the forward ReLU zeroing negative activations).
"""

import torch
import torch.nn as nn
import numpy as np

try:
    from captum.attr import GuidedBackprop as CaptumGuidedBackprop
    _CAPTUM_GB_AVAILABLE = True
except ImportError:
    _CAPTUM_GB_AVAILABLE = False


class GuidedBackpropScratch:
    """
    Guided Backpropagation implemented from scratch with PyTorch backward hooks.

    Usage:
        gbp = GuidedBackpropScratch(model)
        attr = gbp.attribute(image_tensor.unsqueeze(0), target_class=1)
        gbp.remove_hooks()
    """

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                h = module.register_backward_hook(self._guided_relu_backward)
                self.hooks.append(h)

    @staticmethod
    def _guided_relu_backward(module, grad_in, grad_out):
        """
        Guided backprop rule: only propagate non-negative gradients.

        grad_out[0] is the gradient flowing into this ReLU from the layer above.
        Standard ReLU backward: pass grad_out[0] where forward output > 0.
        Guided addition: also clamp grad_out[0] to non-negative values.

        Together these ensure we only backpropagate through neurons that
        were both (a) active in the forward pass and (b) receive a positive
        gradient signal.
        """
        return (torch.clamp(grad_out[0], min=0.0),)

    def attribute(self, image_tensor, target_class):
        """
        Compute Guided Backprop attribution.

        Args:
            image_tensor: (1, C, H, W) — batched input with no grad
            target_class: int

        Returns:
            (H, W) numpy array — positive contributions only
        """
        self.model.eval()
        inp = image_tensor.clone().detach().requires_grad_(True)

        output = self.model(inp)
        self.model.zero_grad()
        output[0, int(target_class)].backward()

        attr = inp.grad.data.squeeze()
        if attr.dim() == 3:
            attr = attr.mean(dim=0)
        attr = torch.clamp(attr, min=0)
        return attr.cpu().numpy()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ---------------------------------------------------------------------------
# Convenience functions used in the notebook
# ---------------------------------------------------------------------------

def compute_guided_backprop_scratch(model, image_tensor, target_class):
    """Guided Backprop from scratch (pure PyTorch hooks)."""
    gbp = GuidedBackpropScratch(model)
    attr = gbp.attribute(image_tensor.unsqueeze(0), target_class)
    gbp.remove_hooks()
    return attr


def compute_guided_backprop_captum(model, image_tensor, target_class):
    """Guided Backprop using Captum (for comparison)."""
    if not _CAPTUM_GB_AVAILABLE:
        raise ImportError("captum is required: pip install captum")
    model.eval()
    gbp = CaptumGuidedBackprop(model)
    inp = image_tensor.unsqueeze(0)
    attr = gbp.attribute(inp, target=int(target_class))
    attr = attr.squeeze().detach().cpu()
    if attr.dim() == 3:
        attr = attr.abs().mean(dim=0)
    return np.maximum(attr.numpy(), 0)


def compare_guided_backprop(model, image_tensor, target_class):
    """
    Compare custom (scratch) vs Captum Guided Backpropagation.

    Returns:
        dict with keys:
            'custom'      — (H, W) normalised attribution (scratch)
            'captum'      — (H, W) normalised attribution (Captum)
            'correlation' — Pearson r between the two (float)
    """
    custom = compute_guided_backprop_scratch(model, image_tensor, target_class)
    captum = compute_guided_backprop_captum(model, image_tensor, target_class)

    def norm01(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-8)

    custom_n = norm01(custom)
    captum_n = norm01(captum)

    corr_matrix = np.corrcoef(custom_n.flatten(), captum_n.flatten())
    corr = float(corr_matrix[0, 1])

    return {
        'custom': custom_n,
        'captum': captum_n,
        'correlation': corr,
    }
