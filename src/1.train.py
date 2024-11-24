from re import L
from config import PATHS, MODEL_PATH, PREDICTIONS_CSV, ADVERSARIAL_PREDICTIONS_CSV
"""
Training script for hate content detection model
"""

import pandas as pd
import tensorflow as tf
from utils import ImageProcessor, SimpleMetalKMeans




def train():
    """Main training function"""
    print("Starting hate content detection model training...")
    
    # Print TensorFlow device info
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Available devices:", physical_devices)
    
    # Initialize processor
    processor = ImageProcessor()
    
    try:
        # Load and process training data
        print("\nLoading training data...")
        train_folder = PATHS['TRAIN_DATA']
        train_images, train_ids = processor.load_images(train_folder)
        
        # Preprocess data
        print("\nPreprocessing training data...")
        train_data = train_images.reshape(train_images.shape[0], -1) / 255.0
        print(f"Training data shape: {train_data.shape}")
        
        # Initialize and train model
        print("\nInitializing model...")
        model = SimpleMetalKMeans(
            n_clusters=2,
            random_state=42,
            batch_size=1024
        )
        
        print("\nTraining model...")
        model.fit(train_data)
        
        # Save the trained model
        print("\nSaving trained model...")
        model.save()
        
        # Process test data
        print("\nProcessing test data...")
        test_folder = PATHS['TEST_DATA']
        test_images, test_ids = processor.load_images(test_folder)
        test_data = test_images.reshape(test_images.shape[0], -1) / 255.0
        
        # Make predictions
        print("\nMaking predictions...")
        predictions = model.predict(test_data)
        
        # Save results
        print("\nSaving results...")
        results_df = pd.DataFrame({
            'image_id': test_ids,
            'prediction': predictions
        })
        results_df.to_csv(PREDICTIONS_CSV, index=False)
        print("\nPredictions saved to PREDICTIONS_CSV")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        raise

if __name__ == "__main__":
    train()