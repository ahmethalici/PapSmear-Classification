# src/models/train_unet.py

import os
import argparse
import pandas as pd
import tensorflow as tf
from pathlib import Path
from omegaconf import OmegaConf

from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from models.model_architectures import build_unet_segmenter, dice_loss

def create_segmentation_dataset(df, config, shuffle=False):
    unet_cfg = config.paper_replication.unet
    def load_patch_and_mask(img_path, mask_path):
        img = tf.cast(tf.io.decode_png(tf.io.read_file(img_path), channels=3), tf.float32) / 255.0
        mask = tf.cast(tf.io.decode_png(tf.io.read_file(mask_path), channels=1), tf.float32) / 255.0
        return img, mask

    dataset = tf.data.Dataset.from_tensor_slices((df['image_path'].values, df['mask_path'].values))
    dataset = dataset.map(load_patch_and_mask, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df), seed=config.paper_replication.seed)
    dataset = dataset.batch(unet_cfg.batch_size)
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

def train_segmenter(config):
    """Main training function for the U-Net segmenter."""
    
    cfg = config.paper_replication
    unet_cfg = cfg.unet
    tf.keras.utils.set_random_seed(cfg.seed)
    
    model_path = Path(unet_cfg.model_path)
    patch_df_path = Path(unet_cfg.patch_df_path)

    if model_path.exists():
        print(f"Found pre-trained segmenter at '{model_path}'. Skipping training.")
        return
        
    if not patch_df_path.exists():
        print(f"ERROR: Patch data CSV not found at '{patch_df_path}'.")
        print("Please run `src/data/generate_patches.py` first.")
        return

    unet_patch_df = pd.read_csv(patch_df_path)
    train_df, val_df = train_test_split(unet_patch_df, test_size=0.2, random_state=cfg.seed)
    
    train_ds = create_segmentation_dataset(train_df, config, shuffle=True)
    val_ds = create_segmentation_dataset(val_df, config)

    model = build_unet_segmenter(input_shape=(unet_cfg.patch_size, unet_cfg.patch_size, 3))
    model.compile(optimizer=Adam(learning_rate=unet_cfg.learning_rate), loss=dice_loss, metrics=['accuracy'])
    model.summary()

    checkpoint = ModelCheckpoint(str(model_path), monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=unet_cfg.early_stopping_patience, restore_best_weights=True)
    
    print("\n--- Starting U-Net training ---")
    model.fit(train_ds, epochs=cfg.epochs, validation_data=val_ds, callbacks=[checkpoint, early_stop])
    print(f"Training complete. Best model saved to {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/config.yml", help="Path to config")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    train_segmenter(config)