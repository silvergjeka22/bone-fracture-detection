# Bone Fracture Detection - Visual Intelligence Project

MSc in Artificial Intelligence - Visual Intelligence  
Academic Year: 2025/2026  
University of Verona

---

## What is This Project?

This project tackles the critical medical challenge of **automatically detecting bone fractures in X-ray images** using deep learning. We compare different neural network architectures and use explainable AI (XAI) techniques to understand *how* and *why* the models make their predictions.

### The Challenge

Medical image analysis requires not just accurate predictions, but also interpretability. Doctors need to understand why an AI system classifies an image as showing a fracture. This project addresses both aspects:

1. **Performance**: Build models that accurately classify bone fractures
2. **Interpretability**: Explain which parts of the image influenced the decision

---

## What We're Doing

### 1. Model Development & Comparison

We implement and compare multiple deep learning architectures:

- **Custom CNN**: A 4-layer convolutional neural network designed specifically for this task
- **Transfer Learning Models**: Pre-trained ResNet and VGG models fine-tuned on bone fracture images
- **Scattering Network (ScatNet)**: A wavelet-based approach using mathematical transforms instead of learned filters

**Goal**: Determine which architecture works best for bone fracture detection and understand why.

### 2. Rigorous Evaluation

We don't just train once and hope for the best. Our evaluation includes:

- **K-Fold Cross-Validation**: Train and validate models multiple times on different data splits to ensure robust performance
- **Multiple Metrics**: Track accuracy, F1 score, precision, and recall
- **Performance Target**: Achieve ≥75% accuracy on unseen test data

**Goal**: Ensure our models generalize well and aren't just memorizing the training data.

### 3. Filter Visualization

Neural networks learn to detect patterns through filters. We visualize:

- **CNN Filters**: What edge detectors, textures, and patterns did the CNN learn?
- **ScatNet Filters**: How do wavelet-based features differ from learned CNN features?

**Goal**: Understand what low-level features each model uses to detect fractures.

### 4. Explainable AI (XAI) Analysis

This is where we answer: *"Why did the model predict a fracture here?"*

We implement and compare **six different XAI methods**:

1. **Gradient-based Attribution**: Which pixels have the strongest influence on the prediction?
2. **Integrated Gradients**: A more robust version that considers the entire path from baseline to input
3. **GradCAM**: Which regions of the image are most important for the decision?
4. **Saliency Maps**: Highlight the most sensitive pixels
5. **DeepLift**: Compare activations to a reference baseline
6. **Custom Implementation**: Build one method from scratch to deeply understand how it works

**Goal**: Generate visual explanations showing which parts of the X-ray led to the fracture prediction, and compare which XAI method provides the most useful insights for medical professionals.

---

## The Dataset

- **Task**: Binary classification (Fractured vs. Non-fractured)
- **Input**: X-ray images of bones
- **Splits**: Training set (with cross-validation) and Test set

---

## Technical Implementation

### Models Architecture

#### Custom CNN
- 4 convolutional blocks with increasing filters [32 → 64 → 128 → 256]
- Batch normalization and max pooling
- Dropout for regularization
- Fully connected classifier

#### Transfer Learning (ResNet18/50, VGG16/19)
- Pre-trained on ImageNet (1M+ images)
- Fine-tuned on bone fracture dataset
- Custom classifier head for binary classification

#### ScatNet (To be implemented)
- Wavelet scattering transform using Kymatio library
- Mathematical feature extraction (no learning required for features)
- Learned classifier on top of scattering coefficients

### Training Strategy

- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Cross-entropy loss
- **Regularization**: Dropout, batch normalization
- **Validation**: 5-fold cross-validation
- **Early Stopping**: Prevent overfitting

### XAI Implementation

Each XAI method generates an **attribution map** that highlights important regions:
- Bright areas = high importance for the prediction
- Dark areas = low importance
- Can be overlaid on original X-ray for interpretation

---

## Getting Started

### Installation

Required libraries:
- **PyTorch**: Deep learning framework
- **Captum**: XAI methods library
- **Kymatio**: Scattering network implementation
- **scikit-learn**: Evaluation metrics and cross-validation
- **matplotlib/seaborn**: Visualization

### Running the Project

1. **Open the notebook**: `bone_fracture_detection.ipynb`
2. **Follow the workflow**:
   - Load and explore the dataset
   - Train models with cross-validation
   - Evaluate on test set
   - Visualize learned filters
   - Generate XAI attribution maps
   - Compare all results

### Using the Modular Code

The project is organized into reusable modules:

```python
# Import models
from src.models import BoneFractureCNN, BoneFractureResNet18

# Import training utilities
from src.training import train_kfold_cv, test_model

# Import visualization
from src.visualization import PlotVisualizer

# Import utilities
from src.utils.device import get_device
from src.utils.helpers import set_seed
```

---

## Expected Results

### Model Performance
- Learning curves showing training/validation progress
- Cross-validation results with mean ± std accuracy
- Test set performance with confusion matrix
- Comparison across all models

### Filter Analysis
- Visualization of what each model "sees"
- Comparison of learned vs. wavelet features
- Insights into feature hierarchies

### XAI Analysis
- Attribution maps for correct predictions
- Attribution maps for incorrect predictions
- Comparison of different XAI methods
- Validation: Custom implementation vs. Captum library

---

## Academic Requirements

This project is part of the Visual Intelligence course:

- **Report**: 6-8 pages (individual) or 8-10 pages (group)
- **Presentation**: 12 minutes (individual) or 15 minutes (group)
- **Language**: English
- **Deliverables**:
  - Working code with all implementations
  - Comprehensive analysis and comparison
  - Visual results and explanations
  - Critical discussion of findings

---

## Key Questions We Answer

1. **Which model architecture works best for bone fracture detection?**
   - Compare CNN, ResNet, VGG, ScatNet
   - Analyze trade-offs: accuracy vs. complexity vs. interpretability

2. **What features do the models learn?**
   - Visualize and interpret learned filters
   - Compare learned features vs. wavelet features

3. **Which XAI method provides the most useful explanations?**
   - Compare attribution quality across 6 methods
   - Evaluate medical interpretability

4. **Can we trust the model's predictions?**
   - Analyze attribution maps for correct/incorrect predictions
   - Identify potential biases or artifacts

5. **How does our custom XAI implementation compare to established libraries?**
   - Validate correctness
   - Understand implementation details

---

## References & Resources

- **PyTorch**: https://pytorch.org/docs/
- **Captum (XAI)**: https://captum.ai/
- **Kymatio (ScatNet)**: https://www.kymat.io/

### Key Papers
- Integrated Gradients: Sundararajan et al. (2017)
- GradCAM: Selvaraju et al. (2017)
- DeepLift: Shrikumar et al. (2017)
- Scattering Networks: Bruna & Mallat (2013)

---

## Contact

**Course Instructors:**
- Prof. Gloria Menegaz: gloria.menegaz@univr.it
- Giorgio Dolci: giorgio.dolci@univr.it

---

## License

This project is developed for academic purposes as part of the Visual Intelligence course at the University of Verona.

---

## Project Organization

The codebase is organized into modular components:

- `src/models/` - Neural network architectures (CNN, ResNet, VGG, ScatNet)
- `src/training/` - Training loops, evaluation, cross-validation
- `src/visualization/` - Plotting and visualization utilities
- `src/xai/` - Explainable AI methods (6 implementations)
- `src/utils/` - Helper functions and device management
- `src/config/` - Model and training configurations
- `outputs/` - Saved models, figures, and results
- `notebooks/` - Experimental notebooks

This modular structure makes it easy to:
- Add new models or XAI methods
- Reuse components across experiments
- Test individual modules
- Maintain and extend the codebase