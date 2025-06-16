# src/models/train_mrf_dcn.py

import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from omegaconf import OmegaConf
import cv2

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness, GaussianNoise

from models.model_architectures import build_mrf_dcn_classifier

def load_sipakmed_for_classification(data_path, label_map):
    """Scans Sipakmed directories for cropped cell images and labels."""

    filepaths, labels = [], []
    print("Searching for cropped cell images for classification...")
    for bmp_path in data_path.glob('im_*/**/*CROPPED*/*.bmp'):
        path_str = str(bmp_path)
        label = 'normal' if 'Parabasal' in path_str or 'Superficial-Intermediate' in path_str or 'Metaplastic' in path_str else 'abnormal'
        filepaths.append(path_str)
        labels.append(label)
    df = pd.DataFrame({'filepath': filepaths, 'label': labels})
    df['label_id'] = df['label'].map(label_map)
    print(f"Found {len(df)} images.")
    return df

def apply_color_preprocessing(image):
    """Applies advanced color and contrast enhancements."""

    img_np = image.numpy().astype(np.uint8)
    img_filtered = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img_filtered, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return tf.convert_to_tensor(img_enhanced, dtype=tf.float32)

@tf.function
def tf_color_preprocessing_wrapper(image, label, img_size):
    """A TensorFlow Function wrapper for the color preprocessing."""

    img_size_tuple = tuple(img_size)
    processed_image, = tf.py_function(lambda img: [apply_color_preprocessing(img)], [image], [tf.float32])
    processed_image.set_shape([*img_size_tuple, 3])
    return processed_image, label

def create_classifier_dataset(df, config, augment=False):
    """Creates a tf.data.Dataset for the classification task."""

    mrf_cfg = config.paper_replication.mrf_dcn
    img_size = tuple(mrf_cfg.img_size)
    
    dataset = tf.data.Dataset.from_tensor_slices((df['filepath'].values, df['label_id'].values))
    dataset = dataset.map(lambda fp, lbl: (tf.image.resize(tf.io.decode_bmp(tf.io.read_file(fp), 3), img_size), lbl), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x, y: tf_color_preprocessing_wrapper(x, y, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment:
        aug_layer = tf.keras.Sequential([
            RandomFlip("horizontal_and_vertical"), RandomRotation(0.4), RandomZoom(0.3),
            RandomContrast(0.3), RandomBrightness(factor=0.2), GaussianNoise(stddev=0.1)
        ], name="data_augmentation")
        dataset = dataset.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        
    dataset = dataset.batch(mrf_cfg.batch_size)
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

def train_classifier(config):
    """Main training function for the MRF-DCN classifier."""
    
    cfg = config.paper_replication
    mrf_cfg = cfg.mrf_dcn
    tf.keras.utils.set_random_seed(cfg.seed)
    
    model_path = Path(mrf_cfg.model_path)
    if model_path.exists():
        print(f"Found pre-trained classifier at '{model_path}'. Skipping training.")
        return

    sipakmed_df = load_sipakmed_for_classification(Path(config.paths.raw_data_dir), dict(cfg.label_map))
    train_val_df, _ = train_test_split(sipakmed_df, test_size=cfg.test_split_size, random_state=cfg.seed, stratify=sipakmed_df['label_id'])
    train_df, val_df = train_test_split(train_val_df, test_size=cfg.validation_split_size, random_state=cfg.seed, stratify=train_val_df['label_id'])
    
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_df['label_id']), y=train_df['label_id'].values)
    class_weight_dict = dict(enumerate(weights))

    train_ds = create_classifier_dataset(train_df, config, augment=True)
    val_ds = create_classifier_dataset(val_df, config)

    model = build_mrf_dcn_classifier(input_shape=tuple(mrf_cfg.img_size) + (3,))
    model.compile(optimizer=Adam(learning_rate=mrf_cfg.learning_rate), loss=BinaryCrossentropy(), metrics=['accuracy'])
    model.summary()

    checkpoint = ModelCheckpoint(str(model_path), monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=mrf_cfg.early_stopping_patience, restore_best_weights=True)

    print("\n--- Starting MRF-DCN training ---")
    model.fit(train_ds, epochs=cfg.epochs, validation_data=val_ds, callbacks=[checkpoint, early_stop], class_weight=class_weight_dict)
    print(f"Training complete. Best model saved to {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/config.yml", help="Path to config")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    train_classifier(config)