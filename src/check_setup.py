"""
Script to verify project setup and paths
"""

import os
from config import PATHS, MODEL_PATH, PREDICTIONS_CSV, ADVERSARIAL_PREDICTIONS_CSV

def check_setup():
    """Check if all necessary directories and files exist"""
    print("Checking project setup...")
    
    # Check directories
    required_dirs = {
        'Training Data': PATHS['TRAIN_DATA'],
        'Test Data': PATHS['TEST_DATA'],
        'Model Directory': PATHS['MODEL'],
        'Perturbed Images': PATHS['PERTURBED'],
        'Results Directory': PATHS['RESULTS']
    }
    
    for name, path in required_dirs.items():
        if not os.path.exists(path):
            print(f"❌ {name} directory missing: {path}")
            os.makedirs(path)
            print(f"   Created directory: {path}")
        else:
            print(f"✓ {name} directory exists: {path}")
    
    # Check if model exists
    if os.path.exists(MODEL_PATH):
        print(f"✓ Model file exists: {MODEL_PATH}")
    else:
        print(f"❌ Model file missing: {MODEL_PATH}")
        print("   Please run training script (1.train.py) first")
    
    # Check for data files
    if os.listdir(PATHS['TRAIN_DATA']):
        print(f"✓ Training data present")
    else:
        print(f"❌ No training data found in {PATHS['TRAIN_DATA']}")
    
    if os.listdir(PATHS['TEST_DATA']):
        print(f"✓ Test data present")
    else:
        print(f"❌ No test data found in {PATHS['TEST_DATA']}")
    
    print("\nSetup check complete!")

if __name__ == "__main__":
    check_setup()