# src/data/generate_patches.py

import os
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

def parse_dats_to_mask(dat_paths, image_shape):
    """Converts .dat contour files into a single binary mask."""
    mask = np.zeros(image_shape, dtype=np.uint8)
    files_to_process = [p for p in dat_paths if 'cyt' in Path(p).name]
    if not files_to_process:
        files_to_process = [p for p in dat_paths if 'nuc' in Path(p).name]

    for dat_path in files_to_process:
        try:
            points = np.loadtxt(dat_path, delimiter=',', dtype=np.int32)
            if points.ndim == 2 and points.shape[1] == 2:
                cv2.drawContours(mask, [points], -1, 255, -1)
        except Exception:
            continue
    return mask

def generate_patches(config):
    """Extracts and saves patches from whole slide images and their masks."""
    print("--- Starting Patch Generation ---")
    cfg = config.paper_replication.unet
    raw_data_path = Path(config.paths.raw_data_dir)
    
    patch_dir = Path(cfg.patch_dir)
    patch_images_dir = patch_dir / 'images'
    patch_masks_dir = patch_dir / 'masks'
    patch_dir.mkdir(parents=True, exist_ok=True)
    patch_images_dir.mkdir(exist_ok=True)
    patch_masks_dir.mkdir(exist_ok=True)

    wsi_data_map = defaultdict(list)
    for dat_path in raw_data_path.glob('im_*/**/*.dat'):
        base_name = dat_path.name.split('_')[0]
        bmp_path = dat_path.parent / f'{base_name}.bmp'
        if bmp_path.exists() and 'CROPPED' not in str(bmp_path):
             wsi_data_map[str(bmp_path)].append(str(dat_path))

    patch_data = []
    patch_counter = 0
    patch_size = cfg.patch_size
    stride = patch_size // 2

    for img_path_str, dat_paths in tqdm(wsi_data_map.items(), desc="Processing WSIs"):
        full_img = cv2.imread(img_path_str)
        if full_img is None: continue
        h, w, _ = full_img.shape
        full_mask = parse_dats_to_mask(dat_paths, (h, w))

        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                img_patch = full_img[y:y+patch_size, x:x+patch_size]
                mask_patch = full_mask[y:y+patch_size, x:x+patch_size]

                if np.mean(mask_patch) < 2.0: continue

                img_patch_path = patch_images_dir / f'patch_{patch_counter}.png'
                mask_patch_path = patch_masks_dir / f'patch_{patch_counter}.png'
                cv2.imwrite(str(img_patch_path), img_patch)
                cv2.imwrite(str(mask_patch_path), mask_patch)
                patch_data.append({'image_path': str(img_patch_path), 'mask_path': str(mask_patch_path)})
                patch_counter += 1

    print(f"\nPatch generation complete. Generated {patch_counter} valid patches.")
    patch_df = pd.DataFrame(patch_data)
    patch_df.to_csv(cfg.patch_df_path, index=False)
    print(f"Patch dataframe saved to {cfg.patch_df_path}")
    
    # After generation, zip the results
    print("\nZipping generated patches for future use...")
    shutil.make_archive(str(cfg.patch_zip_path).replace('.zip',''), 'zip', patch_dir)
    print(f"Patches archived to '{cfg.patch_zip_path}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and zip patches for U-Net training.")
    parser.add_argument("--config", "-c", default="configs/config.yml", help="Path to the config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    
    zip_path = Path(config.paper_replication.unet.patch_zip_path)
    if zip_path.exists():
        print(f"Patch archive already exists at {zip_path}. Unzipping...")
        shutil.unpack_archive(zip_path, Path(config.paper_replication.unet.patch_dir).parent, format='zip')
        print("Unzip complete.")
    elif not list(Path(config.paths.raw_data_dir).glob('im_*/**/*.dat')):
        print("No .dat annotation files found. Cannot generate patches.")
    else:
        print("No patch archive found. Starting generation process...")
        generate_patches(config)