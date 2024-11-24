from config import PATHS, MODEL_PATH, PREDICTIONS_CSV, ADVERSARIAL_PREDICTIONS_CSV
import sys
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def verify_setup():
    """
    Verify that all required packages are properly installed and configured
    """
    print("Python version:", sys.version)
    print("\nTensorFlow version:", tf.__version__)
    print("TensorFlow Metal device:", tf.config.list_physical_devices('GPU'))
    
    print("\nOpenCV version:", cv2.__version__)
    print("NumPy version:", np.__version__)
    print("Pandas version:", pd.__version__)
    
    # Test Metal acceleration
    print("\nTesting Metal acceleration...")
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        print("Matrix multiplication test successful")
    
    print("\nAll components verified successfully!")

if __name__ == "__main__":
    verify_setup()