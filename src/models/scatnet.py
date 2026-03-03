
'''scartNet TODO'''
import torch
import torch.nn as nn

# Uncomment when implementing:
# from kymatio.torch import Scattering2D


class BoneFractureScatNet(nn.Module):
    
    def __init__(self, num_classes=2, J=2, L=8):
        super(BoneFractureScatNet, self).__init__()
 
        raise NotImplementedError(
            "ScatNet implementation is pending. "
            "implement using Kymatio library."
        )