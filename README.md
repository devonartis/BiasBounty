# Hate Content Detection Model
## Bias Bounty 2 - Counterterrorism Challenge

Unsupervised learning model optimized for Apple Silicon to identify extremist content in images. Features comprehensive adversarial testing and Metal GPU acceleration.

## Setup

1. **Create Environment**
```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate hate-detection
```

2. **Directory Structure**
```
.
├── model/                 # Trained model
│   └── kmeans_model.pkl
├── perturbed/            # Adversarial examples
│   └── *.jpg
├── src/                  # Source code
│   ├── utils.py          # Utilities and model class
│   ├── config.py         # Path configurations
│   └── 2.inference.py    # Inference script
├── predictions.csv       # Test predictions
└── adversarial_predictions.csv  # Adversarial predictions
```

## Usage

Run inference:
```bash
python src/2.inference.py
```

This will:
1. Load the trained model
2. Process test images
3. Generate predictions
4. Save results to CSV

## Model Details

1. **Architecture**
   - Unsupervised K-means clustering
   - Metal-optimized implementation
   - Binary classification (0: non-hate, 1: hate)

2. **Features**
   - GPU acceleration for Apple Silicon
   - Parallel image processing
   - Memory-efficient operations

3. **Advanced Features**
   - Multiple adversarial attack strategies
   - Perturbed image generation
   - Robustness analysis

## File Formats

1. **Model**
   - Format: .pkl
   - Location: model/kmeans_model.pkl

2. **Predictions**
   - Format: CSV
   - Columns: image_id, prediction_label
   - Values: 0 (non-hate), 1 (hate)

3. **Adversarial Examples**
   - Format: .jpg
   - Location: perturbed/
   - Naming: adv_[type]_[original_id].jpg

## Dependencies

Key requirements (see environment.yml):
- Python 3.10
- TensorFlow Metal
- OpenCV
- NumPy
- Pandas

## Hardware Requirements

- Apple Silicon Mac (M1/M2/M3)
- 16GB RAM recommended
- macOS 12.0 or later

## Contact

Devon Artis<Troin.Artis@gmail.com>
