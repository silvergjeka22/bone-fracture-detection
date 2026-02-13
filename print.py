import os
import random
import numpy as np
import pandas as pd
from PIL import Image


class DatasetPrinter:
    """
    A utility class for printing dataset statistics and summaries.
    Handles analysis and visualization of image datasets split into train/val/test.
    """

    def __init__(self, base_path=None):
        """
        Initialize the DatasetPrinter.

        Args:
            base_path: Optional base path for the dataset
        """
        self.base_path = base_path

    def print_split_summary(self, base_path, train_path, val_path, test_path):
        """
        Prints the number of images based on train/test/val splits.

        Args:
            base_path: Base directory path
            train_path: Path to training data
            val_path: Path to validation data
            test_path: Path to test data
        """
        for split_name, split_path in [('TRAIN', train_path), ('VAL', val_path), ('TEST', test_path)]:
            print(f"\n{split_name}:")
            if os.path.exists(split_path):
                classes = os.listdir(split_path)
                for cls in classes:
                    cls_path = os.path.join(split_path, cls)
                    if os.path.isdir(cls_path):
                        num_images = len([f for f in os.listdir(cls_path)
                                         if f.endswith(('.png', '.jpg', '.jpeg'))])
                        print(f" {cls}: {num_images} images")

    def analyze_split(self, split_path, split_name):
        """
        Analyze one split (train/val/test) and return statistics.

        Args:
            split_path: Path to the split directory
            split_name: Name of the split (e.g., 'TRAIN', 'VAL', 'TEST')

        Returns:
            Dictionary containing statistics about the split
        """
        stats = {
            'split_name': split_name,
            'classes': {},
            'total': 0,
            'image_sizes': [],
            'widths': [],
            'heights': [],
            'formats': []
        }

        if not os.path.exists(split_path):
            return stats

        print(f"\nAnalyzing {split_name}")

        for class_name in sorted(os.listdir(split_path)):
            class_path = os.path.join(split_path, class_name)

            if os.path.isdir(class_path):
                images = [f for f in os.listdir(class_path)
                         if f.endswith(('.png', '.jpg', '.jpeg'))]

                stats['classes'][class_name] = len(images)
                stats['total'] += len(images)

                # sample images for dimension analysis
                samples = random.sample(images, min(50, len(images)))

                for img_file in samples:
                    try:
                        img_path = os.path.join(class_path, img_file)
                        img = Image.open(img_path)
                        stats['image_sizes'].append(img.size)
                        stats['widths'].append(img.size[0])
                        stats['heights'].append(img.size[1])
                        stats['formats'].append(img.format)
                    except:
                        pass

        return stats

    def print_all_stats(self, all_stats):
        """
        Prints all statistics info based on train/val/test splits.

        Args:
            all_stats: List of statistics dictionaries from analyze_split
        """
        for stats in all_stats:
            if stats['total'] > 0:
                print(f"\n{stats['split_name']}:")
                print(f"  Total images: {stats['total']:,}")

                for cls, count in stats['classes'].items():
                    pct = count / stats['total'] * 100
                    print(f"    • {cls}: {count:,} ({pct:.1f}%)")

                if stats['widths']:
                    print(f"  Image dimensions:")
                    print(f"    Width:  {np.mean(stats['widths']):.0f} ± {np.std(stats['widths']):.0f}")
                    print(f"    Height: {np.mean(stats['heights']):.0f} ± {np.std(stats['heights']):.0f}")

        # total dataset
        total_all = sum(s['total'] for s in all_stats)
        print(f"\nTOTAL DATASET: {total_all:,} images")

    def print_summary_table(self, all_stats):
        """
        Print summary statistics table.

        Args:
            all_stats: List of statistics dictionaries from analyze_split
        """
        summary_data = []
        for stats in all_stats:
            if stats['total'] > 0:
                row = {
                    'Split': stats['split_name'],
                    'Total Images': f"{stats['total']:,}",
                    'Classes': len(stats['classes'])
                }

                if stats['widths']:
                    row['Avg Width'] = f"{np.mean(stats['widths']):.0f}"
                    row['Avg Height'] = f"{np.mean(stats['heights']):.0f}"

                summary_data.append(row)

        df_summary = pd.DataFrame(summary_data)
        print("SUMMARY TABLE")
        print("\n", df_summary.to_string(index=False))

    @staticmethod
    def count_parameters(model):
        """
        Count trainable parameters in a model.

        Args:
            model: PyTorch model

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def print_all_info(self, all_stats, train_path, val_path, test_path):
        """
        Print all information including split summary, stats, and summary table.

        Args:
            all_stats: List of statistics dictionaries from analyze_split
            train_path: Path to training data
            val_path: Path to validation data
            test_path: Path to test data
        """
        self.print_split_summary(self.base_path, train_path, val_path, test_path)
        self.print_all_stats(all_stats)
        self.print_summary_table(all_stats)

    
    def print_detailed_metrics(self, test_labels, test_preds, class_names):
    """
    Print detailed classification metrics including F1, classification report, and confusion matrix.
    
    Args:
        test_labels: True labels
        test_preds: Predicted labels
        class_names: List of class names
    """
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    
    # F1 Score
    f1 = f1_score(test_labels, test_preds, average='binary')
    print("\n" + "="*70)
    print("DETAILED CLASSIFICATION METRICS")
    print("="*70)
    print(f"\nTest F1 Score: {f1:.4f}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    print("\nConfusion Matrix:")
    print(cm)
    print()
    
    # Detailed breakdown
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix Breakdown:")
    print(f"  True Negatives  (TN): {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives  (TP): {tp}")
    
    # Additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nAdditional Metrics:")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print("="*70)