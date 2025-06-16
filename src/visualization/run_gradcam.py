# src/visualization/run_gradcam.py

import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf

from data.make_dataset import load_and_split_data, get_preprocessing_function
from visualization.visualize import get_grad_model, generate_gradcam_heatmap, overlay_heatmap

def run_analysis(config):
    """Loads a model and visualizes Grad-CAM heatmaps on test images."""
    cfg = config.gradcam_analysis
    
    print("Loading Data")
    _, _, _, sipakmed_test_df = load_and_split_data(config)
    
    if sipakmed_test_df.empty:
        print("Test data is empty. Exiting.")
        return

    print(f"Loading Model: {cfg.model_filename}")
    model_path = Path(config.paths.saved_models_dir) / cfg.model_filename
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return
        
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(model_path, compile=False)
    
    print("Creating Grad-CAM Model")
    grad_model = get_grad_model(model, cfg.backbone_name, cfg.last_conv_layer_name)
    if grad_model is None:
        return

    # Get the correct preprocessing function for the model
    preprocessor = get_preprocessing_function(cfg.preproc_fn)
    img_size = tuple(cfg.img_size)

    print(f"Sampling {cfg.num_samples} images from class '{cfg.target_class_name}' ")
    target_label_id = config.data.label_map.get(cfg.target_class_name)
    sample_df = sipakmed_test_df[sipakmed_test_df['label_id'] == target_label_id].sample(
        n=cfg.num_samples, 
        random_state=config.data.seed
    )

    print("Generating and Displaying Visualizations")
    for _, row in sample_df.iterrows():
        img_path = Path(row["filepath"])
        
        # Load and preprocess image
        img = tf.keras.utils.load_img(img_path, target_size=img_size)
        img_array = tf.keras.utils.img_to_array(img)
        preprocessed_img = preprocessor(np.expand_dims(img_array.copy(), axis=0))
        
        # Get prediction and heatmap
        pred_score = model.predict(preprocessed_img, verbose=0)[0, 0]
        pred_class = config.data.class_names[int(pred_score > 0.5)]
        heatmap = generate_gradcam_heatmap(preprocessed_img, grad_model)
        
        # Create overlay
        overlay = overlay_heatmap(img_array, heatmap, alpha=cfg.overlay_alpha)

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title(f"Grad-CAM (Pred: {pred_class} | Score: {pred_score:.2f})")
        plt.axis("off")

        plt.suptitle(f"Grad-CAM on {img_path.name}", fontsize=14)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Grad-CAM analysis on a trained model.")
    parser.add_argument("--config", "-c", default="configs/config.yml", help="Path to config file")
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config)
    run_analysis(config)