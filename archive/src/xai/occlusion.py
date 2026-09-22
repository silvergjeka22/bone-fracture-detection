import torch
from captum.attr import Occlusion

def forward_logits(model, x):
    model.eval()
    with torch.no_grad():
        logits = model(x)   # shape: (1, C)
    return logits

def apply_occlusion(x, top, left, patch_size, baseline=0.0):
    x_occ = x.clone()

    _, _, H, W = x.shape

    bottom = min(top + patch_size, H)
    right = min(left + patch_size, W)

    x_occ[:, :, top:bottom, left:right] = baseline

    return x_occ

import torch

def occlusion_scratch(
    model,
    x,
    patch_size=16,
    stride=8,
    baseline=0.0,
    target_class=1,
    device='cuda'
):

    model = model.to(device)
    x = x.unsqueeze(0).to(device)
    model.eval()

    _, _, H, W = x.shape

    with torch.no_grad():
        logits = model(x)
        original_score = logits[0, target_class]

    heatmap = torch.zeros((H, W), device=device)

    for i in range(0, H, stride):
        for j in range(0, W, stride):

            x_occ = apply_occlusion(x, i, j, patch_size, baseline)

            with torch.no_grad():
                occ_logits = model(x_occ)
                occ_score = occ_logits[0, target_class]

            importance = original_score - occ_score

            i_end = min(i + patch_size, H)
            j_end = min(j + patch_size, W)

            heatmap[i:i_end, j:j_end] = importance.item()

    return heatmap.detach().cpu().numpy()


def captum_occlusion(model, image_tensor, target_class= None ,device='cuda'):

    model.eval()

    # image shape: [1, C, H, W]
    input_image = image_tensor.unsqueeze(0).to(device)
    input_image.requires_grad_()

    # predicted class
    if target_class == None:
        with torch.no_grad():
            pred = model(input_image)
            target_class = pred.argmax(dim=1).item()

    integrated_gradients = Occlusion(model)

    attributions = integrated_gradients.attribute(
        input_image,
        target=target_class,
        baselines=0.0,
        sliding_window_shapes= (3,8,8),
        strides=4
    )
    attributions = attributions.squeeze().detach().cpu().mean(0)
    return attributions.numpy()