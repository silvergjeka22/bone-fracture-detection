import torch
import numpy
from captum.attr import Lime

def compute_feature_mask(inputs, device='cuda'):
    C, H, W = inputs[0].shape
    
    n_rows, n_cols = 14, 14          
    patch_h = H // n_rows         
    patch_w = W // n_cols         
    
    feature_mask = torch.zeros((H, W), dtype=torch.long)
    
    region_id = 0
    for i in range(n_rows):
        for j in range(n_cols):
            h_start = i * patch_h
            w_start = j * patch_w
            h_end = (i + 1) * patch_h
            w_end = (j + 1) * patch_w
    
            feature_mask[h_start:h_end, w_start:w_end] = region_id
            region_id += 1
    
    feature_mask = feature_mask.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
    return feature_mask


def compute_lime_attributions(model, image_tensor, target_class, feature_mask, device='cuda'):

    model.eval()

    # image shape: [1, C, H, W]
    input_image = image_tensor.unsqueeze(0).to(device)

    # predicted class
    with torch.no_grad():
        pred = model(input_image)
        target_class = pred.argmax(dim=1).item()

    lime = Lime(model)

    attributions = lime.attribute(
        input_image,
        target=target_class,
        # baselines=feature_mask,
        feature_mask=feature_mask,
        n_samples=1000,
        # perturbations_per_eval = 200,

    )
    attributions = attributions.squeeze().cpu().detach().mean(0)
    return attributions.numpy()
