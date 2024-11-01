"""
Basic Adversarial Testing Script
"""

import os
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm
import pandas as pd
from utils import ImageProcessor, SimpleMetalKMeans
from config import PATHS, MODEL_PATH, ADVERSARIAL_PREDICTIONS_CSV

class AdversarialGenerator:
    def __init__(self, model_path=MODEL_PATH, epsilon=0.15):
        self.model = SimpleMetalKMeans.load(model_path)
        self.processor = ImageProcessor()
        self.epsilon = epsilon

    def generate_adversarial(self, image):
        """Generate adversarial example using simple gradient-based method"""
        image_flat = image.reshape(1, -1)
        current_pred = self.model.predict(image_flat)[0]
        
        # Get direction towards opposite cluster
        other_cluster = 1 - current_pred
        direction = self.model.centroids[other_cluster].numpy() - image_flat
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        # Generate perturbation
        perturbed = image_flat + self.epsilon * direction
        return np.clip(perturbed, 0, 1).reshape(image.shape)

    def generate_examples(self, input_folder, output_folder):
        """Generate and save adversarial examples"""
        os.makedirs(output_folder, exist_ok=True)
        print(f"\nLoading images from: {input_folder}")
        print(f"Saving perturbed images to: {output_folder}")
        
        # Load images
        images, image_ids = self.processor.load_images(input_folder)
        if len(images) == 0:
            raise ValueError(f"No valid images found in {input_folder}")
            
        # Normalize images
        images_norm = images.astype(np.float32) / 255.0
        
        # Generate adversarial examples
        adversarial_images = []
        adversarial_ids = []
        
        print("\nGenerating adversarial examples...")
        for image, image_id in tqdm(zip(images_norm, image_ids)):
            try:
                # Generate adversarial example
                perturbed = self.generate_adversarial(image)
                
                # Convert back to uint8
                perturbed_uint8 = (perturbed * 255).astype(np.uint8)
                
                # Save perturbed image
                output_path = os.path.join(output_folder, f"adv_{image_id}.jpg")
                cv2.imwrite(output_path, 
                          cv2.cvtColor(perturbed_uint8, cv2.COLOR_RGB2BGR))
                
                adversarial_images.append(perturbed)
                adversarial_ids.append(f"adv_{image_id}")
                
            except Exception as e:
                print(f"\nError processing image {image_id}: {e}")
                continue
        
        return np.array(adversarial_images), adversarial_ids

def main():
    """Main execution function"""
    try:
        print("Starting adversarial example generation...")
        
        # Initialize generator
        generator = AdversarialGenerator(epsilon=0.15)
        
        # Set paths
        test_folder = PATHS['TEST_DATA']
        perturbed_folder = PATHS['PERTURBED']
        
        print(f"Input folder: {test_folder}")
        print(f"Output folder: {perturbed_folder}")
        
        # Generate adversarial examples
        adv_images, adv_ids = generator.generate_examples(test_folder, perturbed_folder)
        
        if len(adv_images) == 0:
            raise ValueError("No adversarial examples were generated successfully")
        
        # Get predictions on adversarial examples
        print("\nGenerating predictions on adversarial examples...")
        adv_data = np.array(adv_images).reshape(len(adv_images), -1)
        adv_predictions = generator.model.predict(adv_data)
        
        # Save results
        results_df = pd.DataFrame({
            'image_id': adv_ids,
            'prediction': adv_predictions
        })
        results_df.to_csv(ADVERSARIAL_PREDICTIONS_CSV, index=False)
        print(f"\nPredictions saved to: {ADVERSARIAL_PREDICTIONS_CSV}")
        
        # Print summary
        print("\nAdversarial Generation Summary:")
        print(f"Total images processed: {len(adv_images)}")
        print("\nPrediction distribution:")
        print(pd.Series(adv_predictions).value_counts())
        
    except Exception as e:
        print(f"\nError in adversarial generation: {e}")
        raise

if __name__ == "__main__":
    main()