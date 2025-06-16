#!/bin/bash
# ==============================================================================
# run_mrf_dcn_unet_pipeline.sh
#
# This script executes the full pipeline for two lightweight models:
#   1. MRF-DCN: A multi-resolution classifier.
#   2. U-Net: A semantic segmenter trained on image patches.
#
# It handles patch generation, training, and evaluation for both.
# ==============================================================================

set -e

echo "=================================================="
echo "    STARTING MRF-DCN & U-NET PIPELINE"
echo "=================================================="

CONFIG_FILE="configs/config.yml"

echo -e "\n[STEP 1/4] Checking for/Generating U-Net patches..."
python3 -m src.data.generate_patches --config $CONFIG_FILE

echo -e "\n[STEP 2/4] Running MRF-DCN classifier training..."
python3 -m src.models.train_mrf_dcn --config $CONFIG_FILE

echo -e "\n[STEP 3/4] Running U-Net segmenter training..."
python3 -m src.models.train_unet --config $CONFIG_FILE

echo -e "\n[STEP 4/4] Running final evaluation for both models..."
python3 -m src.models.evaluate_paper_models --config $CONFIG_FILE
