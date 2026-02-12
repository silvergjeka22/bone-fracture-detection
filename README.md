# Bone Fracture Detection - Visual Intelligence Project

MSc in Artificial Intelligence - Visual Intelligence  
Academic Year: 2025/2026  
University of Verona

## Project Overview

This project implements and compares two deep learning approaches for binary bone fracture classification:
- **2D Convolutional Neural Network (CNN)**
- **Scattering Network (ScatNet)**

The project includes comprehensive model evaluation, filter visualization, and explainable AI (XAI) analysis to understand model decision-making processes.

## Objectives

1. Implement a 2D CNN and ScatNet for binary image classification
2. Compare model performance using k-fold cross-validation
3. Visualize and compare learned filters from both architectures
4. Apply and compare six different XAI methods
5. Implement one XAI method from scratch and validate against Captum library
6. Qualitatively analyze attribution maps for both models

## Dataset

**Task**: Binary classification of bone fracture images  
**Classes**: Fractured vs. Non-fractured bones

The dataset is split into:
- Training set (with k-fold cross-validation)
- Test set

## Requirements

### Python Libraries
- Python 3.8+
- PyTorch
- Captum (for XAI methods)
- Kymatio (for ScatNet implementation)
- NumPy
- Matplotlib
- scikit-learn
- Pillow

### Installation

```bash
pip install torch torchvision
pip install captum
pip install kymatio
pip install numpy matplotlib scikit-learn pillow
```

## Project Structure

```
bone-fracture-detection/
├── bone_fracture_detection.ipynb  # Main implementation notebook
├── plot.py                         # Visualization utilities
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore rules
```

## Implementation Details

### Models

#### CNN Architecture
- Custom 2D Convolutional Neural Network
- Multiple convolutional layers with ReLU activation
- Max pooling for spatial dimension reduction
- Fully connected classifier

#### ScatNet Architecture
- Scattering transform using Kymatio library
- Wavelet-based feature extraction
- Same final classifier as CNN (except input dimensions)

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Target Performance**: ≥75% accuracy on test set

### K-Fold Cross-Validation
- K-fold cross-validation on training set
- Mean accuracy and F1 score computed across folds
- Ensures model generalization ability

### XAI Methods
Six explainable AI methods are applied to both models:
1. Gradient-based attribution
2. Integrated Gradients
3. GradCAM
4. Saliency Maps
5. DeepLift
6. [One method implemented from scratch]

### Filter Visualization
- Extract and visualize filters from first convolutional layer (CNN)
- Extract and visualize scattering filters (ScatNet)
- Compare learned representations

## Usage

### Running the Notebook

1. Open `bone_fracture_detection.ipynb` in Jupyter Notebook or Google Colab
2. Execute cells sequentially:
   - Data loading and preprocessing
   - Model definition and training
   - K-fold cross-validation
   - Filter visualization
   - XAI analysis
   - Results comparison

### Visualization

The `plot.py` module provides utilities for:
- Learning curve visualization
- Filter visualization
- Attribution map overlay
- Comparative analysis plots

## Results

### Model Performance
- Training/validation learning curves
- Mean accuracy and F1 scores from k-fold CV
- Test set performance metrics

### Filter Analysis
- Visualization of learned filters
- Comparison between CNN and ScatNet filters

### XAI Analysis
- Attribution maps for both models
- Comparison of different XAI methods
- Validation of custom implementation vs. Captum

## Report and Presentation

- **Report**: 6-8 pages (individual) or 8-10 pages (group)
- **Presentation**: 12 minutes (individual) or 15 minutes (group)
- **Language**: English
- **Format**: Well-organized sections with clear figures and analysis

## Key Findings

[To be completed after analysis]
- Which model performs better and why
- Which XAI methods provide the most interpretable attributions
- Insights from filter visualization
- Limitations and future work

## References

- PyTorch Documentation: https://pytorch.org/docs/
- Captum Documentation: https://captum.ai/
- Kymatio Documentation: https://www.kymat.io/

## Contact

For questions regarding this project:
- Prof. Gloria Menegaz: gloria.menegaz@univr.it
- Giorgio Dolci: giorgio.dolci@univr.it

## License

This project is developed for academic purposes as part of the Visual Intelligence course at the University of Verona.