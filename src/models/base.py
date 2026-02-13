"""
Base model class for all bone fracture detection models.

This module provides an abstract base class that defines the common interface
for all model architectures in the project.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all bone fracture detection models.
    
    All models should inherit from this class and implement the required methods.
    This ensures a consistent API across different model architectures.
    """
    
    def __init__(self):
        super(BaseModel, self).__init__()
    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        pass
    
    @abstractmethod
    def get_conv1_filters(self):
        """
        Extract first convolutional layer filters for visualization.
        
        Returns:
            Tensor containing the filters from the first convolutional layer
        """
        pass
    
    def get_name(self):
        """
        Get the name of the model.
        
        Returns:
            String name of the model class
        """
        return self.__class__.__name__
    
    def count_parameters(self):
        """
        Count the number of trainable parameters in the model.
        
        Returns:
            Integer count of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_device(self):
        """
        Get the device the model is currently on.
        
        Returns:
            torch.device object
        """
        return next(self.parameters()).device
