# Installation Guide

## System Requirements

### Hardware
- Apple Silicon Mac (M1/M2/M3)
- Minimum 16GB RAM recommended
- SSD storage recommended

### Software
- macOS 12.0 or later
- Python 3.10 or later
- Conda package manager

## Step-by-Step Installation

1. **Install Conda** (if not already installed):
   ```bash
   # Download Miniforge installer for Apple Silicon
   curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
   bash Miniforge3-MacOSX-arm64.sh
   ```

2. **Create Conda Environment**:
   ```bash
conda env create -f environment.yml
conda activate hate-detection
   ```
*** Note: if you choose to not use the environment.yml 

1. **Install TensorFlow Dependencies**:
   ```bash
   conda install -c apple tensorflow-deps
   ```

2. **Install Required Packages**:
   ```bash
   pip install tensorflow-macos
   pip install tensorflow-metal
   pip install -r requirements.txt
   ```

## Dependencies List

```txt
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

## Verification

Run the verification script:
```bash
python verify.py
```

Expected output:
```
GPU acceleration: Available
Metal plugin: Enabled
System memory: Sufficient
All dependencies: Installed
```

## Troubleshooting

### Common Issues

1. **GPU Not Detected**:
   ```bash
   # Reset TensorFlow settings
   conda install -c apple tensorflow-deps --force-reinstall
   pip install tensorflow-macos tensorflow-metal --force-reinstall
   ```

2. **Memory Issues**:
   - Reduce batch size in config
   - Close other applications
   - Check available system memory

3. **Missing Dependencies**:
   ```bash
   pip install -r requirements.txt --upgrade
   ```

### Getting Help

If you encounter issues:
1. Check the error message
2. Consult documentation
3. Contact support with error details

## Additional Resources

- [TensorFlow Metal Documentation](https://developer.apple.com/metal/tensorflow-plugin/)
- [Apple Silicon Developer Guide](https://developer.apple.com/documentation/apple_silicon/)
- [Project Documentation](docs/)