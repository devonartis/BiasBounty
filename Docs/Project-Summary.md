# Hate Content Detection - Advanced Implementation
## Bias Bounty 2 - Counterterrorism Challenge

This implementation provides an advanced solution optimized for Apple Silicon (M1/M2/M3), featuring comprehensive adversarial testing and visualization capabilities.

## Requirements Met

### 1. Basic Requirements ✓
- Unsupervised model (Metal-optimized K-means)
- Binary classification (0: non-hate, 1: hate)
- Test dataset validation
- Required file formats

### 2. Advanced Requirements ✓
- Multiple adversarial attack strategies
- Comprehensive robustness testing
- Perturbed image generation
- Advanced visualization suite

## Project Files

### Core Implementation
1. `utils.py`: Core utilities and model implementation
   - Metal-optimized image processing
   - GPU-accelerated K-means clustering
   - Model persistence handlers

2. `1.train.py`: Training script
   - Trains model on provided dataset
   - Saves model in .pkl format
   - Generates initial predictions

3. `2.inference.py`: Inference script
   - Loads trained model
   - Processes test images
   - Generates prediction CSV

### Advanced Components

4. `3.adversarial.py`: Basic adversarial testing
   - Centroid-based perturbations
   - Noise injection
   - Prediction analysis

5. `4.advance_adversary.py`: Advanced adversarial suite
   - Multiple attack strategies:
     - Centroid-based attack
     - Gaussian noise attack
     - Boundary attack
     - Rotation attack
     - Patch attack
     - Color transformation attack
     - Combined attacks
   - Comprehensive analysis reports
   - Success rate metrics

6. `5.visualize.py`: Visualization suite
   - Cluster analysis
   - Feature importance
   - Sample visualization
   - Stability analysis
   - Interactive plotting

## Apple Silicon Optimizations

1. **Metal GPU Acceleration**
   - TensorFlow Metal plugin integration
   - Optimized memory management
   - Batch processing for efficiency

2. **Performance Enhancements**
   - Multi-threaded image processing
   - Parallel data loading
   - Memory-efficient operations

3. **Hardware-Specific Tuning**
   - Thread count optimization
   - Memory allocation strategies
   - Batch size optimization

## Output Files

1. **Required Files**
   - `model/kmeans_model.pkl`: Trained model
   - `predictions.csv`: Test dataset predictions
   - `adversarial_predictions.csv`: Adversarial example predictions

2. **Additional Files**
   - `Adversarial_data/`: Generated adversarial images
   - `visualization_results/`: Analysis visualizations
   - Detailed performance reports

## Advanced Features

### 1. Adversarial Testing Suite
- Seven different attack strategies
- Customizable attack parameters
- Success rate analysis
- Robustness metrics

### 2. Visualization Capabilities
- PCA and t-SNE visualizations
- Feature importance analysis
- Cluster stability assessment
- Sample image analysis

### 3. Model Analysis Tools
- Cross-validation stability
- Decision boundary analysis
- Feature importance ranking
- Distance distribution analysis

## Dependencies
```
tensorflow-macos
tensorflow-metal
opencv-python
pandas
tqdm
joblib
matplotlib
seaborn
scikit-learn
numpy
```

## Usage Instructions

1. **Environment Setup**:
```bash
conda create -n hate-detection python=3.10
conda activate hate-detection
conda install -c apple tensorflow-deps
pip install -r requirements.txt
```

2. **Training**:
```bash
python 1.train.py
```

3. **Inference**:
```bash
python 2.inference.py
```

4. **Adversarial Testing**:
```bash
python 3.adversarial.py  # Basic testing
python 4.advance_adversary.py  # Advanced suite
```

5. **Visualization**:
```bash
python 5.visualize.py
```

## Innovation Highlights

1. **Metal Optimization**
   - Custom implementation for Apple Silicon
   - Efficient GPU memory usage
   - Parallel processing capabilities

2. **Advanced Testing**
   - Multiple attack strategies
   - Comprehensive robustness analysis
   - Detailed performance metrics

3. **Visualization Suite**
   - Interactive analysis tools
   - Multiple visualization methods
   - Detailed performance reports

4. **Model Improvements**
   - Batch processing
   - Memory efficiency
   - Stability enhancements

## Notes for Evaluators

1. **Hardware Requirements**
   - Apple Silicon Mac (M1/M2/M3)
   - Minimum 16GB RAM recommended
   - SSD storage for better performance

2. **Performance Considerations**
   - GPU acceleration automatic
   - Memory usage optimized
   - Batch size adjustable

3. **Advanced Features Usage**
   - Comprehensive testing suite
   - Detailed analysis capabilities
   - Extensive visualization options

## Contact

For questions or support, please contact [Your Contact Information]
