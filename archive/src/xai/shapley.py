import torch
import numpy
from captum.attr import ShapleyValueSampling

def compute_shapley_attributions(model, image_tensor, target_class, feature_mask, device='cuda'):

    model.eval()

    # image shape: [1, C, H, W]
    input_image = image_tensor.unsqueeze(0).to(device)

    # predicted class
    with torch.no_grad():
        pred = model(input_image)
        target_class = pred.argmax(dim=1).item()

    shapley = ShapleyValueSampling(model)

    attributions = shapley.attribute(
        input_image,
        target=target_class,
        feature_mask=feature_mask,
        n_samples=50,
        perturbations_per_eval = 10,

    )
    attributions = attributions.squeeze().cpu().detach().mean(0)
    return attributions.numpy()