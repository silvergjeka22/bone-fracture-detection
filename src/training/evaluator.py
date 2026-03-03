import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def test_model(model, test_loader, criterion, device):
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


def evaluate_model_batch(model, train_loader, device, model_name="Model"):
    """
    Evaluate a model on a single batch from the train_loader.
    
    Args:
        model: PyTorch model to evaluate
        train_loader: DataLoader to get batch from
        device: Device to run computation on (cpu/cuda)
        model_name: Name of the model for display purposes
    
    Returns:
        tuple: (accuracy, images, labels, predictions)
    """
    # Get a batch from train_loader
    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)
    
    print(f"\n{'='*50}")
    print(f"{model_name} Evaluation")
    print(f"{'='*50}")
    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")
    print(f"Sample labels: {labels[:5]}")
    
    # Calculate accuracy for this batch
    correct = (predictions == labels).sum().item()
    accuracy = correct / labels.size(0)
    print(f"\nBatch accuracy (random init): {accuracy*100:.2f}%")
    
    return accuracy, images, labels, predictions
