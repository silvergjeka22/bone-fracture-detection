class PlotVisualizer:
    
    def __init__(self):
        pass
    
    def plot_split_distribution(self, all_stats, save_path='split_distribution.png'):
        """Plot dataset split distribution (train/val/test) as bar and pie charts."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        splits = [s['split_name'] for s in all_stats if s['total'] > 0]
        totals = [s['total'] for s in all_stats if s['total'] > 0]
        colors_splits = ['#FF6B6B', '#4ECDC4', '#95E1D3']
        
        # Bar chart
        bars = axes[0].bar(splits, totals, color=colors_splits, edgecolor='black', linewidth=2)
        axes[0].set_title('Dataset Split Distribution', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Number of Images', fontweight='bold', fontsize=12)
        axes[0].grid(axis='y', alpha=0.3)
        
        for bar, total in zip(bars, totals):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + max(totals)*0.02,
                        f'{total:,}\n({total/sum(totals)*100:.1f}%)',
                        ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        axes[1].pie(totals, labels=splits, autopct='%1.1f%%', colors=colors_splits,
                   startangle=90, explode=[0.05]*len(splits), shadow=True,
                   textprops={'fontsize': 12, 'fontweight': 'bold'})
        axes[1].set_title('Split Proportion', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {save_path}")
    
    def plot_class_distribution_per_split(self, all_stats, save_path='class_per_split.png'):
        """Plot class distribution for each split (train/val/test)."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, stats in enumerate(all_stats):
            if stats['total'] > 0:
                classes = list(stats['classes'].keys())
                counts = list(stats['classes'].values())
                
                bars = axes[idx].bar(classes, counts, color=['#FF6B6B', '#4ECDC4'],
                                    edgecolor='black', linewidth=2)
                axes[idx].set_title(f"{stats['split_name']} Set", fontsize=14, fontweight='bold')
                axes[idx].set_ylabel('Number of Images', fontweight='bold')
                axes[idx].grid(axis='y', alpha=0.3)
                
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    axes[idx].text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.02,
                                  f'{count}',
                                  ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Class Distribution Across Splits', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {save_path}")
    
    def plot_stacked_distribution(self, all_stats, save_path='stacked_distribution.png'):
        """Plot stacked bar chart showing class distribution across splits."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        all_classes = set()
        for stats in all_stats:
            all_classes.update(stats['classes'].keys())
        all_classes = sorted(list(all_classes))
        
        split_names = [s['split_name'] for s in all_stats if s['total'] > 0]
        class_data = {cls: [] for cls in all_classes}
        
        for stats in all_stats:
            if stats['total'] > 0:
                for cls in all_classes:
                    class_data[cls].append(stats['classes'].get(cls, 0))
        
        x = np.arange(len(split_names))
        width = 0.6
        colors = ['#FF6B6B', '#4ECDC4']
        
        bottom = np.zeros(len(split_names))
        for idx, cls in enumerate(all_classes):
            ax.bar(x, class_data[cls], width, label=cls, bottom=bottom,
                   color=colors[idx], edgecolor='black', linewidth=1.5)
            
            for i, val in enumerate(class_data[cls]):
                if val > 0:
                    ax.text(x[i], bottom[i] + val/2, str(val),
                           ha='center', va='center', fontweight='bold', fontsize=11)
            
            bottom += class_data[cls]
        
        ax.set_title('Stacked Class Distribution', fontsize=16, fontweight='bold')
        ax.set_ylabel('Number of Images', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(split_names, fontweight='bold')
        ax.legend(title='Class', fontsize=11, title_fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {save_path}")
    
    def plot_dimensions_analysis(self, all_stats, save_path='dimensions_analysis.png'):
        """Plot image dimension histograms for each split."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        colors_splits = ['#FF6B6B', '#4ECDC4', '#95E1D3']
        
        for idx, stats in enumerate(all_stats):
            if stats['widths']:
                # Width histogram
                axes[0, idx].hist(stats['widths'], bins=30, color=colors_splits[idx],
                                 alpha=0.7, edgecolor='black')
                axes[0, idx].axvline(np.mean(stats['widths']), color='red',
                                    linestyle='--', linewidth=2,
                                    label=f"Mean: {np.mean(stats['widths']):.0f}")
                axes[0, idx].set_title(f"{stats['split_name']} - Width", fontweight='bold')
                axes[0, idx].set_xlabel('Width (pixels)')
                axes[0, idx].legend()
                axes[0, idx].grid(alpha=0.3)
                
                # Height histogram
                axes[1, idx].hist(stats['heights'], bins=30, color=colors_splits[idx],
                                 alpha=0.7, edgecolor='black')
                axes[1, idx].axvline(np.mean(stats['heights']), color='red',
                                    linestyle='--', linewidth=2,
                                    label=f"Mean: {np.mean(stats['heights']):.0f}")
                axes[1, idx].set_title(f"{stats['split_name']} - Height", fontweight='bold')
                axes[1, idx].set_xlabel('Height (pixels)')
                axes[1, idx].legend()
                axes[1, idx].grid(alpha=0.3)
        
        plt.suptitle('Image Dimensions Across Splits', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {save_path}")
    
    def plot_sample_images(self, train_path, val_path, test_path, save_path='sample_images.png'):
        """Display sample images from each split."""
        fig, axes = plt.subplots(3, 6, figsize=(18, 9))
        
        for row_idx, (split_path, split_name) in enumerate([
            (train_path, 'TRAIN'),
            (val_path, 'VAL'),
            (test_path, 'TEST')
        ]):
            if os.path.exists(split_path):
                classes = [c for c in os.listdir(split_path)
                          if os.path.isdir(os.path.join(split_path, c))]
                
                if classes:
                    class_path = os.path.join(split_path, classes[0])
                    images = [f for f in os.listdir(class_path)
                             if f.endswith(('.png', '.jpg', '.jpeg'))]
                    samples = random.sample(images, min(6, len(images)))
                    
                    for col_idx, img_file in enumerate(samples):
                        img = Image.open(os.path.join(class_path, img_file))
                        axes[row_idx, col_idx].imshow(img, cmap='gray')
                        axes[row_idx, col_idx].axis('off')
                        
                        if col_idx == 0:
                            axes[row_idx, col_idx].set_title(
                                f"{split_name}\n{img.size[0]}x{img.size[1]}",
                                fontweight='bold', fontsize=11)
                        else:
                            axes[row_idx, col_idx].set_title(
                                f"{img.size[0]}x{img.size[1]}", fontsize=9)
        
        plt.suptitle('Sample Images from Each Split', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {save_path}")
    
    def plot_class_balance(self, all_stats, save_path='class_balance.png'):
        """Plot class balance ratios across splits."""
        fig, ax = plt.subplots(figsize=(12, 6))
        colors_splits = ['#FF6B6B', '#4ECDC4', '#95E1D3']
        
        balance_data = []
        for stats in all_stats:
            if stats['classes']:
                counts = list(stats['classes'].values())
                if len(counts) >= 2:
                    balance_ratio = min(counts) / max(counts)
                    balance_data.append({
                        'split': stats['split_name'],
                        'ratio': balance_ratio,
                        'status': 'Balanced' if balance_ratio > 0.8 else 'Imbalanced'
                    })
        
        if balance_data:
            splits_bal = [d['split'] for d in balance_data]
            ratios = [d['ratio'] for d in balance_data]
            
            bars = ax.bar(splits_bal, ratios, color=colors_splits[:len(splits_bal)],
                          edgecolor='black', linewidth=2)
            ax.axhline(0.8, color='red', linestyle='--', linewidth=2,
                       label='Balance Threshold (0.8)')
            ax.set_ylim(0, 1.1)
            ax.set_title('Class Balance Across Splits', fontsize=16, fontweight='bold')
            ax.set_ylabel('Balance Ratio (min/max)', fontweight='bold', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            ax.legend()
            
            for bar, ratio, data in zip(bars, ratios, balance_data):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                       f'{ratio:.3f}\n{data["status"]}',
                       ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {save_path}")
    
    def generate_all_plots(self, all_stats, train_path, val_path, test_path):
        """Generate all dataset visualization plots at once."""
        self.plot_split_distribution(all_stats)
        self.plot_class_distribution_per_split(all_stats)
        self.plot_stacked_distribution(all_stats)
        self.plot_dimensions_analysis(all_stats)
        self.plot_sample_images(train_path, val_path, test_path)
        self.plot_class_balance(all_stats)
    
    def visualize_batch(self, train_loader, train_dataset, num_images=8, figsize=(15, 7)):
        """Visualize a batch of images from the training DataLoader."""
        images, labels = next(iter(train_loader))
        
        print(f"\nBatch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        
        rows = 2
        cols = num_images // rows
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()
        
        for idx in range(num_images):
            img = images[idx].permute(1, 2, 0).cpu().numpy()
            img = img * 0.5 + 0.5
            img = np.clip(img, 0, 1)
            
            axes[idx].imshow(img)
            axes[idx].set_title(f"Label: {train_dataset.class_names[labels[idx]]}",
                               fontweight='bold')
            axes[idx].axis('off')
        
        plt.suptitle('Sample Batch from Training DataLoader', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def visualize_filters(self, model, layer_name='conv1', save_path='cnn_filters.png'):
        """Visualize filters from the first convolutional layer."""
        filters = model.get_conv1_filters().cpu().numpy()
        
        num_filters = filters.shape[0]
        num_channels = filters.shape[1]
        
        print(f"\nFilter shape: {filters.shape}")
        print(f"Number of filters: {num_filters}")
        
        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(min(32, num_filters)):
            filter_img = filters[i].mean(axis=0)
            filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-8)
            
            axes[i].imshow(filter_img, cmap='viridis')
            axes[i].set_title(f'Filter {i+1}', fontsize=8)
            axes[i].axis('off')
        
        plt.suptitle('CNN First Layer Filters (Conv1)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_kfold_history(self, fold_histories, save_path='cnn_kfold_training.png'):
        """Plot training history for all folds."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        
        # Plot Loss
        for i, history in enumerate(fold_histories):
            epochs = range(1, len(history['train_loss']) + 1)
            axes[0].plot(epochs, history['train_loss'],
                        linestyle='-', alpha=0.6, label=f'Fold {i+1} Train')
            axes[0].plot(epochs, history['val_loss'],
                        linestyle='--', alpha=0.6, label=f'Fold {i+1} Val')
        
        axes[0].set_xlabel('Epoch', fontweight='bold', fontsize=12)
        axes[0].set_ylabel('Loss', fontweight='bold', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontweight='bold', fontsize=14)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[0].grid(alpha=0.3)
        
        # Plot Accuracy
        for i, history in enumerate(fold_histories):
            epochs = range(1, len(history['train_acc']) + 1)
            axes[1].plot(epochs, history['train_acc'],
                        linestyle='-', alpha=0.6, label=f'Fold {i+1} Train')
            axes[1].plot(epochs, history['val_acc'],
                        linestyle='--', alpha=0.6, label=f'Fold {i+1} Val')
        
        axes[1].set_xlabel('Epoch', fontweight='bold', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontweight='bold', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontweight='bold', fontsize=14)
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\nSaved: {save_path}")
