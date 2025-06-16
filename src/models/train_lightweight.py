# src/models/train_lightweight.py

import os
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import tensorflow as tf
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from data.make_dataset import load_and_split_data, get_class_weights, create_dataset
from models.model_architectures import build_base_model

def train_lightweight_models(config):

    # load data
    train_df, val_df, _, _ = load_and_split_data(config)
    class_weight_dict = get_class_weights(train_df)
    
    # create the save directory
    models_save_dir = Path(config.paths.saved_models_dir)
    models_save_dir.mkdir(parents=True, exist_ok=True)
    
    # loop through each color variant and model
    for variant in config.variants:
        variant_name = variant.name
        print(f"\n{'-'*80}\nPIPELINE FOR VARIANT: {variant_name.upper()}\n{'-'*80}")
        
        for model_conf in config.models:
            model_name = model_conf.name
            model_name_variant = f"{model_name}_{variant_name}"
            
            finetuned_model_path = models_save_dir / f'{model_name_variant}_best_finetuned.keras' #no .h5 plox
            
            if finetuned_model_path.exists():
                print(f"\n--- SKIPPING: Found existing fine-tuned model for {model_name_variant} ---")
                continue
            
            print(f"\n--- Starting Pipeline for: {model_name_variant} ---")

            # Create datasets for this specific model and variant
            train_ds = create_dataset(train_df, config, shuffle=True, augment=True, 
                                      preprocess_fn_name=model_conf.preproc_fn, 
                                      is_grayscale=variant.is_grayscale, use_color_pp=variant.use_color_pp)
            val_ds = create_dataset(val_df, config, 
                                    preprocess_fn_name=model_conf.preproc_fn,
                                    is_grayscale=variant.is_grayscale, use_color_pp=variant.use_color_pp)

            # Build and compile model
            model = build_base_model(model_conf, config.training, config.data)
            model.compile(optimizer=Nadam(learning_rate=config.training.optimal_hyperparameters.learning_rate),
                          loss=BinaryCrossentropy(), metrics=['accuracy'])
            model.summary()

            # Callbacks
            initial_model_path = models_save_dir / f'{model_name_variant}_best.keras'
            checkpoint = ModelCheckpoint(str(initial_model_path), monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
            early_stop = EarlyStopping(monitor='val_loss', patience=config.training.early_stopping_patience, restore_best_weights=True, verbose=1)
            log_dir = Path(config.paths.log_dir) / f"{model_name_variant}_initial"
            tensorboard_cb = TensorBoard(log_dir=log_dir)

            # Feature Extraction
            print(f"\n[Stage 1/2] Initial Training for {model_name_variant}...")
            history_initial = model.fit(
                train_ds, epochs=config.training.initial_train_epochs, validation_data=val_ds,
                callbacks=[checkpoint, early_stop, tensorboard_cb], class_weight=class_weight_dict, verbose=1
            )

            # finetune
            print(f"\n[Stage 2/2] Fine-Tuning for {model_name_variant}...")
            
            model = tf.keras.models.load_model(str(initial_model_path))
            
            # unreeze layaers
            base_model = model.layers[1]
            base_model.trainable = True
            fine_tune_from = int(len(base_model.layers) * (1 - config.training.finetune_unfreeze_percent))
            for layer in base_model.layers[:fine_tune_from]:
                layer.trainable = False
            
            model.compile(optimizer=Nadam(learning_rate=config.training.optimal_hyperparameters.learning_rate / 10),
                          loss=BinaryCrossentropy(), metrics=['accuracy'])
            
            finetune_checkpoint = ModelCheckpoint(str(finetuned_model_path), monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
            log_dir_finetune = Path(config.paths.log_dir) / f"{model_name_variant}_finetune"
            tensorboard_cb_finetune = TensorBoard(log_dir=log_dir_finetune)
            start_epoch = len(history_initial.epoch)

            model.fit(
                train_ds, epochs=start_epoch + config.training.finetune_epochs, initial_epoch=start_epoch,
                validation_data=val_ds, callbacks=[finetune_checkpoint, early_stop, tensorboard_cb_finetune],
                class_weight=class_weight_dict, verbose=1
            )
            print(f"\n--- Finished pipeline for {model_name_variant} ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        # The default is the BASE config.
        default="configs/config.yml",
        help="Path to the base or override config file."
    )
    args = parser.parse_args()

    base_config_path = "configs/config.yml"
    print(f"--- Loading base configuration from: {base_config_path} ---")
    config = OmegaConf.load(base_config_path)

    if args.config != base_config_path:
        override_config_path = args.config
        print(f"--- Loading and merging override config: {override_config_path} ---")
        override_conf = OmegaConf.load(override_config_path)
        # The override config values will replace the base config values.
        config = OmegaConf.merge(config, override_conf)

    # print("--- Final effective configuration ---")

    train_lightweight_models(config)
