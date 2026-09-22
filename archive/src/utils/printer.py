import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, classification_report, confusion_matrix

class DatasetPrinter:
    @staticmethod
    def print_split_summary(base_path, train_path, val_path, test_path):
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

    @staticmethod
    def analyze_split(split_path, split_name):
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

    @staticmethod
    def print_all_stats(all_stats):
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

    @staticmethod 
    def print_summary_table(all_stats):
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
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def print_all_info(base_path, all_stats, train_path, val_path, test_path):
        DatasetPrinter.print_split_summary(base_path, train_path, val_path, test_path)
        DatasetPrinter.print_all_stats(all_stats)
        DatasetPrinter.print_summary_table(all_stats)

    @staticmethod
    def print_detailed_metrics(test_labels, test_preds, class_names):

        # F1 Score
        f1 = f1_score(test_labels, test_preds, average='binary')
        print("DETAILED CLASSIFICATION METRICS")
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