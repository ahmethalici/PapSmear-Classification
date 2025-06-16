# src/data/make_dataset.py

import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# data loading and splitting

def find_image_paths_by_type(data_path, label_map):
    image_paths, labels = [], []
    for ext in ['*.bmp', '*.BMP']:
        for file in data_path.rglob(ext):
            parent_folder = file.parent.name.lower()
            if 'carcinoma' in parent_folder or 'dysplastic' in parent_folder:
                labels.append('abnormal')
                image_paths.append(str(file))
            elif 'normal' in parent_folder:
                labels.append('normal')
                image_paths.append(str(file))
    df = pd.DataFrame({'filepath': image_paths, 'label': labels})
    if not df.empty:
        df['label_id'] = df['label'].map(label_map).astype(int)
    return df

def load_and_split_data(config):
    data_path = Path(config.paths.raw_data_dir)
    seed = config.data.seed
    label_map = dict(config.data.label_map)

    print("--- Loading Herlev Dataset ---")
    herlev_train_path = data_path / 'HerlevData' / 'train'
    herlev_test_path = data_path / 'HerlevData' / 'test'
    herlev_train_df = find_image_paths_by_type(herlev_train_path, label_map)
    herlev_test_df = find_image_paths_by_type(herlev_test_path, label_map)
    print(f"Loaded {len(herlev_train_df)} Herlev train and {len(herlev_test_df)} test images.")

    print("\n--- Loading and Splitting Sipakmed Dataset ---")
    sipakmed_paths = glob.glob(str(data_path / 'im_*' / '*' / 'CROPPED' / '*.bmp'))
    sipakmed_labels = ['normal' if 'im_Parabasal' in p or 'im_Superficial-Intermediate' in p else 'abnormal' for p in sipakmed_paths]
    sipakmed_full_df = pd.DataFrame({'filepath': sipakmed_paths, 'label': sipakmed_labels})
    sipakmed_full_df['label_id'] = sipakmed_full_df['label'].map(label_map).astype(int)
    print(f"Loaded {len(sipakmed_full_df)} total Sipakmed images.")

    sip_train_val_df, sip_test_df = train_test_split(
        sipakmed_full_df, test_size=config.data.sipakmed_split.test_size, random_state=seed, stratify=sipakmed_full_df['label_id']
    )
    sip_train_df, sip_val_df = train_test_split(
        sip_train_val_df, test_size=config.data.sipakmed_split.validation_size, random_state=seed, stratify=sip_train_val_df['label_id']
    )

    final_train_df = pd.concat([herlev_train_df, sip_train_df], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"\nFinal Combined Training Set: {len(final_train_df)} images")
    print(f"Final Validation Set (Sipakmed only): {len(sip_val_df)} images")
    print(f"Final Herlev Test Set: {len(herlev_test_df)} images")
    print(f"Final Sipakmed Test Set: {len(sip_test_df)} images")
    return final_train_df, sip_val_df, herlev_test_df, sip_test_df

def get_class_weights(train_df):
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_df['label_id']),
        y=train_df['label_id'].values
    )
    return dict(enumerate(class_weights))

# Preprocessing and tf.data.Dataset pipeline

def get_preprocessing_function(model_preproc_name):
    preproc_name_lower = model_preproc_name.lower()
    if preproc_name_lower == "vgg16":
        return tf.keras.applications.vgg16.preprocess_input
    elif preproc_name_lower == "xception":
        return tf.keras.applications.xception.preprocess_input
    elif preproc_name_lower == "efficientnet":
        return tf.keras.applications.efficientnet.preprocess_input
    elif preproc_name_lower == "inception_resnet_v2":
        return tf.keras.applications.inception_resnet_v2.preprocess_input
    elif preproc_name_lower == "densenet":
        return tf.keras.applications.densenet.preprocess_input
    elif preproc_name_lower == "nasnet":
        return tf.keras.applications.nasnet.preprocess_input
    else:
        # Generic case: just rescale to [0,1] .
        return lambda x: x / 255.0

def apply_color_preprocessing(image):
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
    image, = tf.py_function(lambda img: [apply_color_preprocessing(img)], [image], [tf.float32])
    image.set_shape([*img_size, 3]) # nvm
    return image, label

def get_data_augmentation_layer():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.5),
        tf.keras.layers.RandomZoom(0.3),
        tf.keras.layers.RandomContrast(0.4),
        tf.keras.layers.RandomBrightness(factor=0.3),
        tf.keras.layers.GaussianNoise(stddev=0.1)
    ], name="data_augmentation")

def load_and_resize(filepath, label, img_size):
    img = tf.io.read_file(filepath)
    img = tf.io.decode_bmp(img, channels=3)
    img = tf.image.resize(img, img_size)
    return img, label

def create_dataset(df, config, shuffle=False, augment=False, preprocess_fn_name=None, is_grayscale=False, use_color_pp=False):
    img_size = tuple(config.data.img_size)
    
    dataset = tf.data.Dataset.from_tensor_slices((df['filepath'].values, df['label_id'].values))
    dataset = dataset.map(lambda x, y: load_and_resize(x, y, img_size), num_parallel_calls=tf.data.AUTOTUNE)

    if use_color_pp:
        dataset = dataset.map(lambda x, y: tf_color_preprocessing_wrapper(x, y, img_size), num_parallel_calls=tf.data.AUTOTUNE)

    if is_grayscale:
        dataset = dataset.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda x, y: (tf.image.grayscale_to_rgb(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        aug_layer = get_data_augmentation_layer()
        dataset = dataset.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df), seed=config.data.seed)

    dataset = dataset.batch(config.data.batch_size)

    if preprocess_fn_name:
        preprocess_fn = get_preprocessing_function(preprocess_fn_name)
        dataset = dataset.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)