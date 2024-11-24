"""
Configuration settings for the project
"""

import os

# Get absolute path to project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths relative to root
PATHS = {
    'TRAIN_DATA': os.path.join(ROOT_DIR, 'Training_data'),
    'TEST_DATA': os.path.join(ROOT_DIR, 'Test_data'),
    'PERTURBED': os.path.join(ROOT_DIR, 'perturbed'),
    'MODEL': os.path.join(ROOT_DIR, 'model'),
    'RESULTS': os.path.join(ROOT_DIR, 'adversarial_results')
}

# Model file
MODEL_PATH = os.path.join(PATHS['MODEL'], 'kmeans_model.pkl')

# Results files
PREDICTIONS_CSV = os.path.join(ROOT_DIR, 'predictions.csv')
ADVERSARIAL_PREDICTIONS_CSV = os.path.join(ROOT_DIR, 'adversarial_predictions.csv')

# Create necessary directories
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)