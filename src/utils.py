"""
Core utilities for hate content detection model
"""

import os
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import joblib
from config import MODEL_PATH

def generate_image_ids(image_folder):
    """
    Official image ID generator from competition instructions
    """
    image_ids = []
    for image_file in os.listdir(image_folder):
        # Get the file name without the extension
        image_id = os.path.splitext(image_file)[0]
        image_ids.append(image_id)
    return image_ids

class ImageProcessor:
    """Handles image loading and preprocessing"""
    def __init__(self, size=(64, 64)):
        self.size = size
        self.max_workers = 8

    def process_image(self, file_path):
        """Process a single image"""
        try:
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        return None

    def load_images(self, folder):
        """Load and process images"""
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Directory not found: {folder}")
            
        files = [f for f in os.listdir(folder) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not files:
            raise ValueError(f"No valid images found in {folder}")
            
        images = []
        image_ids = []
        
        print(f"\nProcessing {len(files)} images from {folder}")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for file in files:
                future = executor.submit(
                    self.process_image, 
                    os.path.join(folder, file)
                )
                futures.append((future, file))

            for future, file in tqdm(futures, desc="Loading images", unit="image"):
                img = future.result()
                if img is not None:
                    images.append(img)
                    image_ids.append(os.path.splitext(file)[0])

        print(f"Successfully loaded {len(images)} images")
        return np.array(images, dtype=np.float32), image_ids

class SimpleMetalKMeans:
    """K-means implementation optimized for Metal GPU"""
    def __init__(self, n_clusters=2, random_state=42, max_iter=100, batch_size=1024):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.centroids = None

    def compute_distances(self, X, centroids):
        """Compute distances in batches"""
        n_samples = len(X)
        distances = []
        
        for i in range(0, n_samples, self.batch_size):
            batch = X[i:i + self.batch_size]
            batch_distances = tf.reduce_sum(
                tf.square(
                    tf.expand_dims(batch, 1) - tf.expand_dims(centroids, 0)
                ),
                axis=2
            )
            distances.append(batch_distances)
            
        return tf.concat(distances, axis=0)

    def fit(self, X):
        """Fit K-means to the data"""
        print("\nStarting K-means clustering...")
        X = tf.constant(X, dtype=tf.float32)
        
        np.random.seed(self.random_state)
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = tf.Variable(tf.gather(X, idx))
        
        for iteration in range(self.max_iter):
            distances = self.compute_distances(X, self.centroids)
            assignments = tf.argmin(distances, axis=1)
            
            new_centroids = []
            for k in range(self.n_clusters):
                mask = tf.cast(tf.equal(assignments, k), tf.float32)
                total = tf.reduce_sum(
                    tf.multiply(tf.expand_dims(mask, 1), X),
                    axis=0
                )
                count = tf.reduce_sum(mask) + tf.keras.backend.epsilon()
                new_centroids.append(total / count)
            
            new_centroids = tf.stack(new_centroids)
            
            if tf.reduce_all(tf.abs(new_centroids - self.centroids) < 1e-6):
                print(f"Converged after {iteration + 1} iterations")
                break
                
            self.centroids.assign(new_centroids)
            
            if (iteration + 1) % 10 == 0:
                print(f"Completed iteration {iteration + 1}/{self.max_iter}")
        
        return self

    def predict(self, X):
        """Predict cluster assignments"""
        if self.centroids is None:
            raise ValueError("Model must be fitted before making predictions")
            
        X = tf.constant(X, dtype=tf.float32)
        distances = self.compute_distances(X, self.centroids)
        return tf.argmin(distances, axis=1).numpy()

    def save(self, filepath=None):
        """Save the trained model"""
        if filepath is None:
            filepath = MODEL_PATH
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_state = {
            'centroids': self.centroids.numpy() if self.centroids is not None else None,
            'n_clusters': self.n_clusters,
            'random_state': self.random_state,
            'max_iter': self.max_iter,
            'batch_size': self.batch_size
        }
        joblib.dump(model_state, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath=None):
        """Load a trained model"""
        if filepath is None:
            filepath = MODEL_PATH
            
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        model_state = joblib.load(filepath)
        model = cls(
            n_clusters=model_state['n_clusters'],
            random_state=model_state['random_state'],
            max_iter=model_state['max_iter'],
            batch_size=model_state['batch_size']
        )
        if model_state['centroids'] is not None:
            model.centroids = tf.Variable(model_state['centroids'])
        print(f"Model loaded from {filepath}")
        return model