"""
Model evaluation utilities.

This module contains functions for testing models and computing detailed metrics.
"""

import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def test_model(model, test_loader, criterion, device):
    """
    Test model on test set and return detailed metrics.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to test on (cpu/cuda)
        
    Returns:
        test_loss: Average test loss
        test_acc: Test accuracy
        test_preds: List of predictions
        test_labels: List of true labels
        metrics: Dictionary with precision, recall, f1
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = running_loss / total
    test_acc = correct / total
    
    metrics = {
        'f1': f1_score(all_labels, all_preds, average='binary'),
        'precision': precision_score(all_labels, all_preds, average='binary'),
        'recall': recall_score(all_labels, all_preds, average='binary')
    }
    
    return test_loss, test_acc, all_preds, all_labels, metrics
