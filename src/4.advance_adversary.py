"""
Advanced Adversarial Testing Suite with Multiple Attack Strategies
"""

# Standard imports
import os
import numpy as np
import cv2
from tqdm import tqdm
from datetime import datetime

# ML related imports
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Local imports
from utils import ImageProcessor, SimpleMetalKMeans
from config import PATHS, MODEL_PATH, ADVERSARIAL_PREDICTIONS_CSV

# Configure directories
OUTPUT_DIR = os.path.join(PATHS['RESULTS'], 'analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

class AdvancedAdversarialGenerator:
    """Advanced adversarial example generator with multiple attack strategies"""
    
    def __init__(self, model_path=MODEL_PATH):
        """Initialize generator with model and attack types"""
        self.model = SimpleMetalKMeans.load(model_path)
        self.processor = ImageProcessor()
        
        # Define available attack types
        self.attack_types = {
            'centroid': self._centroid_attack,
            'noise': self._noise_attack,
            'boundary': self._boundary_attack,
            'rotation': self._rotation_attack,
            'patch': self._patch_attack,
            'color': self._color_attack,
            'combined': self._combined_attack
        }
    
    def _centroid_attack(self, image, epsilon=0.15):
        """Move image towards opposite centroid"""
        image_flat = image.reshape(1, -1)
        current_pred = self.model.predict(image_flat)[0]
        other_cluster = 1 - current_pred
        direction = self.model.centroids[other_cluster].numpy() - image_flat
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        return np.clip(image_flat + epsilon * direction, 0, 1).reshape(image.shape)

    def _noise_attack(self, image, epsilon=0.15):
        """Add random noise"""
        noise = np.random.normal(0, epsilon, image.shape)
        return np.clip(image + noise, 0, 1)

    def _boundary_attack(self, image, steps=5):
        """Binary search along decision boundary"""
        image_flat = image.reshape(1, -1)
        current_pred = self.model.predict(image_flat)[0]
        other_centroid = self.model.centroids[1 - current_pred].numpy()
        
        alpha = 0
        beta = 1
        best_pert = image
        
        for _ in range(steps):
            mid = (alpha + beta) / 2
            interpolated = (1 - mid) * image + mid * other_centroid.reshape(image.shape)
            if self.model.predict(interpolated.reshape(1, -1))[0] == current_pred:
                alpha = mid
            else:
                beta = mid
                best_pert = interpolated
        
        return np.clip(best_pert, 0, 1)

    def _rotation_attack(self, image, max_angle=30):
        """Rotate image"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        angle = np.random.uniform(-max_angle, max_angle)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return np.clip(rotated, 0, 1)

    def _patch_attack(self, image, patch_size=0.2):
        """Add adversarial patch"""
        h, w = image.shape[:2]
        ph = int(h * patch_size)
        pw = int(w * patch_size)
        
        # Create random patch
        patch = np.random.uniform(0, 1, (ph, pw, 3))
        
        # Random position
        x = np.random.randint(0, w - pw)
        y = np.random.randint(0, h - ph)
        
        perturbed = image.copy()
        perturbed[y:y+ph, x:x+pw] = patch
        return np.clip(perturbed, 0, 1)

    def _color_attack(self, image, intensity=0.15):
        """Modify color channels"""
        perturbed = image.copy()
        channel = np.random.randint(0, 3)
        perturbed[:, :, channel] = np.clip(
            perturbed[:, :, channel] + np.random.uniform(-intensity, intensity),
            0, 1
        )
        return perturbed

    def _combined_attack(self, image):
        """Combine multiple attacks"""
        attacks = [
            lambda: self._centroid_attack(image, epsilon=0.1),
            lambda: self._noise_attack(image, epsilon=0.1),
            lambda: self._color_attack(image, intensity=0.1),
            lambda: self._rotation_attack(image, max_angle=15)
        ]
        
        # Apply random combination
        num_attacks = np.random.randint(2, len(attacks) + 1)
        selected_attacks = np.random.choice(attacks, num_attacks, replace=False)
        
        perturbed = image.copy()
        for attack in selected_attacks:
            perturbed = attack()
        
        return np.clip(perturbed, 0, 1)

    def generate_adversarial_examples(self, input_folder, output_folder):
        """Generate adversarial examples using all attack types"""
        os.makedirs(output_folder, exist_ok=True)
        print(f"\nLoading images from: {input_folder}")
        print(f"Saving perturbed images to: {output_folder}")
        
        # Load and normalize images
        images, image_ids = self.processor.load_images(input_folder)
        if len(images) == 0:
            raise ValueError(f"No valid images found in {input_folder}")
        images_norm = images.astype(np.float32) / 255.0
        
        results = {
            'original_images': [],
            'perturbed_images': [],
            'original_preds': [],
            'perturbed_preds': [],
            'attack_types': [],
            'success': [],
            'image_ids': []
        }
        
        # Generate adversarial examples
        for image, image_id in tqdm(zip(images_norm, image_ids), 
                                  total=len(images_norm),
                                  desc="Generating adversarial examples"):
            orig_pred = self.model.predict(image.reshape(1, -1))[0]
            
            for attack_name, attack_fn in self.attack_types.items():
                try:
                    # Generate perturbed image
                    perturbed = attack_fn(image)
                    
                    # Get prediction
                    pert_pred = self.model.predict(perturbed.reshape(1, -1))[0]
                    
                    # Save image
                    output_path = os.path.join(output_folder, 
                                             f"adv_{attack_name}_{image_id}.jpg")
                    cv2.imwrite(output_path, 
                              cv2.cvtColor((perturbed * 255).astype(np.uint8),
                                         cv2.COLOR_RGB2BGR))
                    
                    # Store results
                    results['original_images'].append(image)
                    results['perturbed_images'].append(perturbed)
                    results['original_preds'].append(orig_pred)
                    results['perturbed_preds'].append(pert_pred)
                    results['attack_types'].append(attack_name)
                    results['success'].append(orig_pred != pert_pred)
                    results['image_ids'].append(f"adv_{attack_name}_{image_id}")
                    
                except Exception as e:
                    print(f"\nError with {attack_name} attack on {image_id}: {e}")
                    continue
        
        if not results['image_ids']:
            raise ValueError("No adversarial examples were generated successfully")
            
        return results

class AdversarialAnalyzer:
    """Comprehensive analysis of adversarial attacks"""
    def __init__(self, results):
        self.results = results
        
    def generate_report(self):
        """Generate comprehensive analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create report dataframe
        df = pd.DataFrame({
            'image_id': self.results['image_ids'],
            'attack_type': self.results['attack_types'],
            'original_pred': self.results['original_preds'],
            'perturbed_pred': self.results['perturbed_preds'],
            'success': self.results['success']
        })
        
        # Save detailed results
        detailed_results_path = os.path.join(OUTPUT_DIR, 
                                           f'detailed_results_{timestamp}.csv')
        df.to_csv(detailed_results_path, index=False)
        print(f"Detailed results saved to: {detailed_results_path}")
        
        # Generate attack effectiveness analysis
        attack_success = df.groupby('attack_type')['success'].agg(['count', 'sum', 'mean'])
        attack_success.columns = ['Total', 'Successful', 'Success_Rate']
        
        effectiveness_path = os.path.join(OUTPUT_DIR, 
                                        f'attack_effectiveness_{timestamp}.csv')
        attack_success.to_csv(effectiveness_path)
        print(f"Effectiveness analysis saved to: {effectiveness_path}")
        
        # Plot success rates
        plt.figure(figsize=(12, 6))
        sns.barplot(x=attack_success.index, y='Success_Rate', data=attack_success)
        plt.title('Attack Success Rates')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        success_rates_path = os.path.join(OUTPUT_DIR, 
                                        f'success_rates_{timestamp}.png')
        plt.savefig(success_rates_path)
        plt.close()
        print(f"Success rates plot saved to: {success_rates_path}")
        
        # Generate confusion matrices for each attack
        for attack in df['attack_type'].unique():
            attack_data = df[df['attack_type'] == attack]
            cm = confusion_matrix(attack_data['original_pred'], 
                                attack_data['perturbed_pred'])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', 
                       xticklabels=['Non-Hate', 'Hate'],
                       yticklabels=['Non-Hate', 'Hate'])
            plt.title(f'Confusion Matrix - {attack} Attack')
            plt.ylabel('Original Prediction')
            plt.xlabel('Adversarial Prediction')
            plt.tight_layout()
            
            cm_path = os.path.join(OUTPUT_DIR, 
                                 f'confusion_matrix_{attack}_{timestamp}.png')
            plt.savefig(cm_path)
            plt.close()
            print(f"Confusion matrix for {attack} attack saved to: {cm_path}")
            
        # Save main predictions to competition format
        competition_df = df[['image_id', 'perturbed_pred']].copy()
        competition_df.columns = ['image_id', 'prediction']
        competition_df.to_csv(ADVERSARIAL_PREDICTIONS_CSV, index=False)
        print(f"\nCompetition format predictions saved to: {ADVERSARIAL_PREDICTIONS_CSV}")
            
        return df, attack_success

def main():
    """Main execution function"""
    try:
        print("Starting advanced adversarial testing...")
        print(f"Results will be saved to: {OUTPUT_DIR}")
        
        # Initialize generator
        generator = AdvancedAdversarialGenerator(model_path=MODEL_PATH)
        
        # Generate adversarial examples
        test_folder = PATHS['TEST_DATA']
        perturbed_folder = PATHS['PERTURBED']
        print(f"\nInput folder: {test_folder}")
        print(f"Output folder: {perturbed_folder}")
        
        results = generator.generate_adversarial_examples(test_folder, perturbed_folder)
        
        # Analyze results
        analyzer = AdversarialAnalyzer(results)
        df, attack_success = analyzer.generate_report()
        
        # Print summary
        print("\nAdversarial Testing Summary:")
        print("\nAttack Effectiveness:")
        print(attack_success)
        
        print(f"\nAll results have been saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\nError in adversarial testing: {e}")
        raise

if __name__ == "__main__":
    main()