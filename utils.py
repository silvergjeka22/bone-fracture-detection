import torch
import torch.nn as nn
import numpy as np
import time
import copy
from sklearn.model_selection import KFold
from torch.utils.data import Subset


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.

    Args:
        model: PyTorch model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cpu/cuda)

    Returns:
        epoch_loss: Average loss for the epoch
        epoch_acc: Accuracy for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """
    Validate for one epoch.

    Args:
        model: PyTorch model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on (cpu/cuda)

    Returns:
        epoch_loss: Average loss for the epoch
        epoch_acc: Accuracy for the epoch
        all_preds: List of all predictions
        all_labels: List of all labels
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, all_preds, all_labels


def train_kfold_cv(model_class, train_dataset, k_folds=5, num_epochs=30,
                   batch_size=32, learning_rate=0.001, device='cpu'):
    """
    Train model with k-fold cross-validation.

    Args:
        model_class: Model class to instantiate (not an instance)
        train_dataset: Training dataset
        k_folds: Number of folds for cross-validation
        num_epochs: Number of epochs per fold
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on (cpu/cuda)

    Returns:
        fold_results: List of results for each fold
        fold_histories: List of training histories for each fold
        mean_acc: Mean validation accuracy across folds
        std_acc: Standard deviation of validation accuracy
    """

    # Prepare for k-fold
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Store results
    fold_results = []
    fold_histories = []

    # Get all indices
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))

    # K-Fold CV
    for fold, (train_ids, val_ids) in enumerate(kfold.split(indices)):

        print(f"FOLD {fold + 1}/{k_folds}")
        print(f"Train size: {len(train_ids)}, Val size: {len(val_ids)}")

        # Create data subsets
        train_subsampler = Subset(train_dataset, train_ids)
        val_subsampler = Subset(train_dataset, val_ids)

        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_subsampler, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_subsampler, batch_size=batch_size, shuffle=False
        )

        # Initialize model for this fold
        model = model_class(num_classes=2, dropout_rate=0.5)
        model = model.to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        best_val_acc = 0.0
        best_model_wts = copy.deepcopy(model.state_dict())

        # Training loop
        for epoch in range(num_epochs):
            start_time = time.time()

            # Train
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

            # Validate
            val_loss, val_acc, _, _ = validate_epoch(model, val_loader, criterion, device)

            # Update scheduler
            scheduler.step(val_loss)

            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch [{epoch+1:2d}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"Time: {epoch_time:.1f}s")

        # Load best model weights
        model.load_state_dict(best_model_wts)

        # Store fold results
        fold_results.append({
            'fold': fold + 1,
            'best_val_acc': best_val_acc,
            'final_train_acc': history['train_acc'][-1],
            'model': model
        })
        fold_histories.append(history)

        print(f"\nFold {fold + 1} Best Val Accuracy: {best_val_acc:.4f}")

    # Calculate mean results
    mean_acc = np.mean([r['best_val_acc'] for r in fold_results])
    std_acc = np.std([r['best_val_acc'] for r in fold_results])

    print("K-FOLD CROSS-VALIDATION RESULTS")
    for result in fold_results:
        print(f"Fold {result['fold']}: Val Accuracy = {result['best_val_acc']:.4f}")
    print(f"\nMean Val Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print("="*70)

    return fold_results, fold_histories, mean_acc, std_acc


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
    from sklearn.metrics import f1_score, precision_score, recall_score
    
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