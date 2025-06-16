# src/models/evaluate_models.py

import argparse
import os
from pathlib import Path
from omegaconf import OmegaConf
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from data.make_dataset import load_and_split_data, create_dataset
from visualization.visualize import plot_confusion_matrix

def evaluate_single_model(model_path, test_ds, model_name, class_names):
    """
    Loads and evaluates a single model, then prints and plots results.
    """

    print(f"\n--- Evaluating: {model_name} ---")

    try:
        model = tf.keras.models.load_model(str(model_path), compile=False)
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return

    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"  - Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    y_true = np.concatenate([y for _, y in test_ds], axis=0).flatten()
    y_pred_proba = model.predict(test_ds, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    plot_confusion_matrix(y_true, y_pred, class_names, f'Confusion Matrix for {model_name}')

def evaluate_all_models(config):
    # Load test dataframes
    _, _, herlev_test_df, sipakmed_test_df = load_and_split_data(config)
    models_save_dir = Path(config.paths.saved_models_dir)
    class_names = list(config.data.class_names)

    evaluation_tasks = []
    
    # add lightweight models to the evaluation queue
    for variant in config.variants:
        for model_conf in config.models:
            task = {
                "name": f"{model_conf.name} ({variant.name})",
                "path": models_save_dir / f"{model_conf.name}_{variant.name}_best_finetuned.keras",
                "preproc_fn": model_conf.preproc_fn,
                "is_grayscale": variant.is_grayscale,
                "use_color_pp": variant.use_color_pp
            }
            evaluation_tasks.append(task)
            
    # add ensemble models to the evaluation queue
    for variant in config.variants:
        for ensemble_type in ["Averaging", "Stacking"]:
            task = {
                "name": f"{ensemble_type} Ensemble ({variant.name})",
                "path": models_save_dir / f"{ensemble_type}_Ensemble_{variant.name}_best.keras",
                "preproc_fn": None, # Ensembles use generic data
                "is_grayscale": variant.is_grayscale,
                "use_color_pp": variant.use_color_pp
            }
            evaluation_tasks.append(task)

    # run evaluation loop
    for task in evaluation_tasks:
        if not task['path'].exists():
            print(f"\nSKIPPING: Model file not found at '{task['path']}'")
            continue
            
        print("\n" + "#"*35 + f" FINAL RESULTS FOR: {task['name']} " + "#"*35)
        
        # Create test datasets for this specific model configuration
        sipakmed_ds = create_dataset(
            sipakmed_test_df, config, preprocess_fn_name=task["preproc_fn"], 
            is_grayscale=task["is_grayscale"], use_color_pp=task["use_color_pp"]
        )
        herlev_ds = create_dataset(
            herlev_test_df, config, preprocess_fn_name=task["preproc_fn"],
            is_grayscale=task["is_grayscale"], use_color_pp=task["use_color_pp"]
        )
        
        # Evaluate on both test sets
        evaluate_single_model(task['path'], sipakmed_ds, f"{task['name']} on Sipakmed Test", class_names)
        evaluate_single_model(task['path'], herlev_ds, f"{task['name']} on Herlev Test", class_names)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/config.yml",
        help="Path to the base or override config file."
    )
    args = parser.parse_args()

    # 1. Always load the main base configuration file first.
    base_config_path = "configs/config.yml"
    print(f"--- Loading base configuration from: {base_config_path} ---")
    config = OmegaConf.load(base_config_path)

    # 2. If a different config is specified, load it and merge it.
    if args.config != base_config_path:
        override_config_path = args.config
        print(f"--- Loading and merging override config: {override_config_path} ---")
        override_conf = OmegaConf.load(override_config_path)
        config = OmegaConf.merge(config, override_conf)

    # Call the main function for the script
    evaluate_all_models(config)
