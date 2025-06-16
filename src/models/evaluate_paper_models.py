# src/models/evaluate_paper_models.py

import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from models.model_architectures import dice_loss, dice_score
from models.train_mrf_dcn import load_sipakmed_for_classification, create_classifier_dataset
from models.train_unet import create_segmentation_dataset

def evaluate_classifier(config):
    cfg = config.paper_replication
    mrf_cfg = cfg.mrf_dcn
    model_path = Path(mrf_cfg.model_path)
    
    print("\n" + "="*50 + f"\n--- Evaluating Classifier: {model_path.name} ---\n" + "="*50)
    if not model_path.exists():
        print("!!! SKIPPING: Model file not found. !!!")
        return

    sipakmed_df = load_sipakmed_for_classification(Path(config.paths.raw_data_dir), dict(cfg.label_map))
    train_val_df, test_df = train_test_split(sipakmed_df, test_size=cfg.test_split_size, random_state=cfg.seed, stratify=sipakmed_df['label_id'])
    
    test_ds = create_classifier_dataset(test_df, config)
    
    model = load_model(str(model_path))
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    print(f"\n- Test Accuracy: {accuracy:.4f}\n- Test Loss:     {loss:.4f}")
    
    y_true = np.concatenate([y for _, y in test_ds]).flatten()
    y_pred = (model.predict(test_ds).flatten() > 0.5).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=list(cfg.class_names), digits=4))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cfg.class_names, yticklabels=cfg.class_names)
    plt.title(f'Confusion Matrix for {model_path.name}'); plt.xlabel('Predicted'); plt.ylabel('True'); plt.show()

def evaluate_segmenter(config, num_examples=5):
    cfg = config.paper_replication
    unet_cfg = cfg.unet
    model_path = Path(unet_cfg.model_path)
    
    print("\n" + "="*50 + f"\n--- Evaluating Segmenter: {model_path.name} ---\n" + "="*50)
    if not model_path.exists():
        print("!!! SKIPPING: Model file not found. !!!")
        return

    patch_df = pd.read_csv(unet_cfg.patch_df_path)
    _, test_patch_df = train_test_split(patch_df, test_size=0.2, random_state=cfg.seed + 1)
    test_ds_seg = create_segmentation_dataset(test_patch_df, config)

    model = load_model(str(model_path), custom_objects={'dice_loss': dice_loss})

    all_scores = [dice_score(m, model.predict(i, verbose=0)) for i, m in tqdm(test_ds_seg, desc="Calculating Dice Score")]
    print(f"\n- Average Dice Score on Test Set: {np.mean(all_scores):.4f}")

    print(f"\nVisualizing {num_examples} Test Predictions:")
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 5 * num_examples))
    fig.tight_layout(pad=3.0)
    for i, (img, mask) in enumerate(test_ds_seg.unbatch().take(num_examples)):
        pred_mask = model.predict(tf.expand_dims(img, 0), verbose=0)[0]
        score = dice_score(mask, pred_mask).numpy()
        axes[i, 0].imshow(img); axes[i, 0].set_title("Input Patch")
        axes[i, 1].imshow(mask, cmap='gray'); axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 2].imshow(pred_mask, cmap='gray'); axes[i, 2].set_title(f"Predicted Mask\nDice: {score:.4f}")
        for ax in axes[i]: ax.axis('off')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/config.yml", help="Path to config")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    
    evaluate_classifier(config)
    
    if Path(config.paper_replication.unet.patch_df_path).exists():
        evaluate_segmenter(config)
    else:
        print("\nSKIPPING Segmenter Evaluation: No patch data was available.")