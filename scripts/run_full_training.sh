#!/bin/bash

# ==============================================================================
# run_full_training.sh
#
# This script executes the FULL training and evaluation pipeline.
# It uses the main configuration file (configs/config.yml) to train all models
# on the complete dataset and produce the final results.
#
# NOTE: This will be time-consuming and requires the full dataset.
#
# Usage:
#   bash scripts/run_full_training.sh
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

echo "=================================================="
echo "        STARTING FULL TRAINING PIPELINE"
echo "=================================================="

# Define the main configuration file
CONFIG_FILE="configs/config.yml"

echo -e "\n[FULL PIPELINE STEP 1/3] Running lightweight model training..."
echo "This will take a significant amount of time."
python3 -m src.models.train_lightweight --config $CONFIG_FILE

echo -e "\n[FULL PIPELINE STEP 2/3] Running ensemble model training..."
python3 -m src.models.train_ensembles --config $CONFIG_FILE

echo -e "\n[FULL PIPELINE STEP 3/3] Running final evaluation..."
python3 -m src.models.evaluate_models --config $CONFIG_FILE

echo -e "\n=================================================="
echo "      FULL TRAINING PIPELINE COMPLETED SUCCESSFULLY!"
echo "=================================================="