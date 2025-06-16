# src/models/train_ensembles.py

import argparse
from pathlib import Path
from omegaconf import OmegaConf
import tensorflow as tf
import os
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from data.make_dataset import load_and_split_data, get_class_weights, create_dataset, get_preprocessing_function
from models.model_architectures import build_averaging_ensemble, build_stacking_ensemble

def train_ensembles(config):
    # load data
    train_df, val_df, _, _ = load_and_split_data(config)
    class_weight_dict = get_class_weights(train_df)
    
    models_save_dir = Path(config.paths.saved_models_dir)

    for variant in config.variants:
        variant_name = variant.name
        print(f"\n{'-'*80}\nBUILDING ENSEMBLES FOR VARIANT: {variant_name.upper()}\n{'-'*80}")
        
        # load all available fine-tuned base models for this variant
        base_models = []
        preproc_fns = []
        for model_conf in config.models:
            model_path = models_save_dir / f"{model_conf.name}_{variant_name}_best_finetuned.keras"
            if model_path.exists():
                print(f"Loading base model: {model_path.name}")
                base_models.append(tf.keras.models.load_model(str(model_path)))
                preproc_fns.append(get_preprocessing_function(model_conf.preproc_fn))
            else:
                print(f"WARNING: Base model not found, skipping: {model_path.name}")
        
        if len(base_models) < 2:
            print(f"\n--- SKIPPING ensemble for {variant_name}: Fewer than 2 base models available. ---")
            continue

        # Averaging ensemble
        avg_ensemble_path = models_save_dir / f'Averaging_Ensemble_{variant_name}_best.keras'
        if not avg_ensemble_path.exists():
            print("\n--- Building and Saving Averaging Ensemble ---")
            avg_ensemble_model = build_averaging_ensemble(base_models, preproc_fns, variant_name, config.data)
            avg_ensemble_model.compile(loss=BinaryCrossentropy(), metrics=['accuracy'])
            avg_ensemble_model.save(str(avg_ensemble_path))
            print(f"Averaging Ensemble ({variant_name}) created and saved.")
        else:
            print("\n--- SKIPPING: Found existing Averaging Ensemble ---")

        # Sstacking ensemble
        stacking_ensemble_path = models_save_dir / f'Stacking_Ensemble_{variant_name}_best.keras'
        if not stacking_ensemble_path.exists():
            print("\n--- Building and Training Stacking Ensemble ---")
            stacking_model = build_stacking_ensemble(base_models, preproc_fns, variant_name, config.training, config.data)
            stacking_model.compile(
                optimizer=Nadam(learning_rate=config.training.optimal_hyperparameters.learning_rate),
                loss=BinaryCrossentropy(),
                metrics=['accuracy']
            )
            stacking_model.summary()

            # generic datasets for ensemble training
            train_ds_generic = create_dataset(train_df, config, shuffle=True, augment=True, 
                                              is_grayscale=variant.is_grayscale, use_color_pp=variant.use_color_pp)
            val_ds_generic = create_dataset(val_df, config, 
                                            is_grayscale=variant.is_grayscale, use_color_pp=variant.use_color_pp)

            # callbacks
            stack_checkpoint = ModelCheckpoint(str(stacking_ensemble_path), monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
            early_stop_ensemble = EarlyStopping(monitor='val_loss', patience=config.training.ensemble_early_stopping_patience, restore_best_weights=True, verbose=1)
            log_dir = Path(config.paths.log_dir) / f"Stacking_Ensemble_{variant_name}"
            tensorboard_cb = TensorBoard(log_dir=log_dir)

            stacking_model.fit(
                train_ds_generic, epochs=config.training.ensemble_train_epochs, validation_data=val_ds_generic,
                callbacks=[stack_checkpoint, early_stop_ensemble, tensorboard_cb], class_weight=class_weight_dict, verbose=1
            )
        else:
            print("\n--- SKIPPING: Found existing Stacking Ensemble ---")

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
    train_ensembles(config)
