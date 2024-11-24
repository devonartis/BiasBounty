from config import PATHS, MODEL_PATH, PREDICTIONS_CSV, ADVERSARIAL_PREDICTIONS_CSV
"""
Inference script for hate content detection model
"""

import pandas as pd
from utils import ImageProcessor, SimpleMetalKMeans


def run_inference(image_folder, model_path="model/kmeans_model.pkl"):
    """
    Run inference on a folder of images using a trained model.

    Args:
        image_folder (str): Path to folder containing images
        model_path (str): Path to saved model

    Returns:
        pandas.DataFrame: Predictions for each image
    """
    print("Starting inference process...")

    try:
        # Load the trained model
        print("\nLoading trained model...")
        model = SimpleMetalKMeans.load(model_path)

        # Initialize image processor
        processor = ImageProcessor()

        # Load and process images
        print("\nProcessing images...")
        images, image_ids = processor.load_images(image_folder)

        # Preprocess data
        print("\nPreprocessing images...")
        data = images.reshape(images.shape[0], -1) / 255.0
        print(f"Data shape: {data.shape}")

        # Make predictions
        print("\nGenerating predictions...")
        predictions = model.predict(data)

        # Create results DataFrame
        results_df = pd.DataFrame({"image_id": image_ids, "prediction": predictions})

        return results_df

    except Exception as e:
        print(f"\nError during inference: {e}")
        raise


def main():
    """Main function for inference"""
    try:
        # Specify the folder containing images for inference
        inference_folder = "./Test_data"  # Change this to your inference folder

        # Run inference
        results_df = run_inference(inference_folder)

        # Save results
        output_file = "inference_predictions.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nPredictions saved to {output_file}")

        # Print summary
        print("\nPrediction Summary:")
        print(f"Total images processed: {len(results_df)}")
        print("Predictions distribution:")
        print(results_df["prediction"].value_counts())

    except Exception as e:
        print(f"\nError in main: {e}")
        raise


if __name__ == "__main__":
    main()
