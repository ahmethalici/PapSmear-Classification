# create_fake_dataset.py
import os
from pathlib import Path
import numpy as np
from PIL import Image

def create_dummy_bmp(filepath, size=(10, 10)):
    """Creates a small, black BMP image."""
    try:
        im = Image.new('RGB', size, 'black')
        im.save(filepath, 'BMP')
    except Exception as e:
        print(f"Could not create dummy file {filepath}: {e}")

print("Creating fake dataset structure in data/raw/...")
base_path = Path("data/raw/")

structure = {
    # Herlev Data
    "HerlevData/train/normal_in_situ_carcinoma": 2,
    "HerlevData/train/normal_intermediate": 2,
    "HerlevData/test/normal_columnar": 2,
    "HerlevData/test/severe_dysplastic": 2,
    
    "im_Abnormal/class_1/CROPPED": 5, # abnormal
    "im_Carcinoma-in-situ/class_2/CROPPED": 5, # abnormal
    "im_Parabasal/class_3/CROPPED": 5, # normal
    "im_Superficial-Intermediate/class_4/CROPPED": 5, # normal
}

for dir_path, num_files in structure.items():
    full_path = base_path / dir_path
    print(f"Creating directory: {full_path}")
    full_path.mkdir(parents=True, exist_ok=True)
    for i in range(num_files):
        dummy_file = full_path / f"dummy_image_{i}.bmp"
        create_dummy_bmp(dummy_file)

print("\nIMPROVED fake dataset created successfully.")
print("You can now run the pipeline for a dry run.")
