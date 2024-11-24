from config import PATHS, MODEL_PATH, PREDICTIONS_CSV, ADVERSARIAL_PREDICTIONS_CSV
"""
Interactive Visualization Suite for Model Analysis
Provides real-time visual insights with proper display handling
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils import ImageProcessor, SimpleMetalKMeans
import pandas as pd
from datetime import datetime
import cv2
from tqdm import tqdm

# Set basic style parameters
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.facecolor'] = '#f8f8f8'

class ModelVisualizer:
    """Interactive visualization toolkit for model analysis"""
    
    def __init__(self, model_path='model/kmeans_model.pkl', output_dir='visualization_results'):
        self.model = SimpleMetalKMeans.load(model_path)
        self.processor = ImageProcessor()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def show_and_save(self, name, show=True):
        """Helper to show and save plots"""
        plt.tight_layout()
        # Save the plot
        save_path = os.path.join(self.output_dir, f"{name}_{self.timestamp}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        if show:
            plt.show()
        else:
            plt.close()
        
        print(f"Plot saved to: {save_path}")

    def plot_clusters(self, data, predictions, show=True):
        """Interactive cluster visualization"""
        # Create figure with two subplots
        fig = plt.figure(figsize=(20, 8))
        
        # PCA Plot
        plt.subplot(1, 2, 1)
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], 
                            c=predictions, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('Cluster Distribution (PCA)', fontsize=14, pad=20)
        plt.xlabel('First Principal Component', fontsize=12)
        plt.ylabel('Second Principal Component', fontsize=12)
        
        # t-SNE Plot
        plt.subplot(1, 2, 2)
        tsne = TSNE(n_components=2, random_state=42)
        data_tsne = tsne.fit_transform(data)
        scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], 
                            c=predictions, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('Cluster Distribution (t-SNE)', fontsize=14, pad=20)
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        
        self.show_and_save('cluster_distributions', show)
        return data_2d, data_tsne

    def plot_sample_images(self, folder, n_samples=5, show=True):
        """Display sample images from each cluster"""
        images, image_ids = self.processor.load_images(folder)
        data = images.reshape(images.shape[0], -1) / 255.0
        predictions = self.model.predict(data)
        
        # Create subplot grid
        fig, axes = plt.subplots(self.model.n_clusters, n_samples, 
                                figsize=(15, 6))
        fig.suptitle('Sample Images from Each Cluster', fontsize=16, y=1.05)
        
        for cluster in range(self.model.n_clusters):
            cluster_indices = np.where(predictions == cluster)[0]
            if len(cluster_indices) >= n_samples:
                sample_indices = np.random.choice(cluster_indices, 
                                                n_samples, replace=False)
                
                for i, idx in enumerate(sample_indices):
                    axes[cluster, i].imshow(images[idx])
                    axes[cluster, i].set_title(f'Cluster {cluster}', fontsize=10)
                    axes[cluster, i].axis('off')
        
        self.show_and_save('sample_images', show)

    def plot_feature_importance(self, data, show=True):
        """Analyze and visualize feature importance"""
        pca = PCA()
        pca.fit(data)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 8))
        
        # Explained variance plot
        plt.subplot(1, 2, 1)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(cumsum, linewidth=2, color='blue')
        plt.xlabel('Number of Components', fontsize=12)
        plt.ylabel('Cumulative Explained Variance', fontsize=12)
        plt.title('Explained Variance Ratio', fontsize=14)
        
        # Top components visualization
        plt.subplot(1, 2, 2)
        plt.bar(range(10), pca.explained_variance_ratio_[:10] * 100, 
               color='blue', alpha=0.7)
        plt.xlabel('Principal Component', fontsize=12)
        plt.ylabel('Explained Variance (%)', fontsize=12)
        plt.title('Top 10 Principal Components', fontsize=14)
        
        self.show_and_save('feature_importance', show)

    def plot_stability_analysis(self, data, n_runs=5, show=True):
        """Analyze and visualize model stability"""
        predictions_all = []
        
        print("Running stability analysis...")
        for i in tqdm(range(n_runs)):
            model = SimpleMetalKMeans(random_state=i)
            model.fit(data)
            predictions_all.append(model.predict(data))
            
        # Calculate agreement matrix
        agreement_matrix = np.zeros((n_runs, n_runs))
        for i in range(n_runs):
            for j in range(n_runs):
                agreement_matrix[i, j] = np.mean(
                    predictions_all[i] == predictions_all[j])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(agreement_matrix, annot=True, cmap='viridis', 
                   vmin=0, vmax=1, square=True)
        plt.title('Model Stability Analysis', fontsize=14)
        plt.xlabel('Run Number', fontsize=12)
        plt.ylabel('Run Number', fontsize=12)
        
        self.show_and_save('stability_analysis', show)
        return agreement_matrix

    def plot_distance_distribution(self, data, show=True):
        """Plot distribution of distances to centroids"""
        distances = self.model.compute_distances(data, self.model.centroids)
        predictions = self.model.predict(data)
        
        plt.figure(figsize=(12, 6))
        for i in range(self.model.n_clusters):
            cluster_distances = distances.numpy()[predictions == i][:, i]
            plt.hist(cluster_distances, bins=50, alpha=0.5, 
                    label=f'Cluster {i}')
        
        plt.title('Distance Distribution to Centroids', fontsize=14)
        plt.xlabel('Distance', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend()
        
        self.show_and_save('distance_distribution', show)

    def generate_report(self, train_folder, interactive=True):
        """Generate comprehensive visualization report"""
        print("Loading and processing data...")
        train_images, train_ids = self.processor.load_images(train_folder)
        train_data = train_images.reshape(train_images.shape[0], -1) / 255.0
        train_predictions = self.model.predict(train_data)
        
        print("\n1. Generating cluster visualizations...")
        self.plot_clusters(train_data, train_predictions, show=interactive)
        
        print("\n2. Showing sample images...")
        self.plot_sample_images(train_folder, show=interactive)
        
        print("\n3. Analyzing feature importance...")
        self.plot_feature_importance(train_data, show=interactive)
        
        print("\n4. Performing stability analysis...")
        self.plot_stability_analysis(train_data, show=interactive)
        
        print("\n5. Analyzing distance distributions...")
        self.plot_distance_distribution(train_data, show=interactive)
        
        print(f"\nVisualization report generated in {self.output_dir}")
        if not interactive:
            print("All plots have been saved to files.")

def main():
    """Main execution function"""
    try:
        # Ask user for visualization preference
        while True:
            choice = input("Do you want to display plots interactively? (yes/no): ").lower()
            if choice in ['yes', 'no']:
                break
            print("Please enter 'yes' or 'no'")
        
        interactive = (choice == 'yes')
        
        # Create visualizer and generate report
        visualizer = ModelVisualizer()
        visualizer.generate_report(PATHS['TRAIN_DATA'], interactive=interactive)
        
    except Exception as e:
        print(f"\nError during visualization: {e}")
        raise
    finally:
        # Clean up all plots
        plt.close('all')

if __name__ == "__main__":
    main()