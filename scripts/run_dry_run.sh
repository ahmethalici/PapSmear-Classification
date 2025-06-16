#!/bin/bash

# ==============================================================================
# run_dry_run.sh
#
# This script executes a fast, lightweight "dry run" of the entire pipeline.
# It uses the test configuration to ensure all scripts run without errors.
# It's perfect for verifying your environment setup.
#
# Usage:
#   bash scripts/run_dry_run.sh
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

echo "=================================================="
echo "          STARTING PIPELINE DRY RUN"
echo "=================================================="

# Define the test configuration file
CONFIG_FILE="configs/config_test.yml"

echo -e "\n[DRY RUN STEP 1/4] Creating fake dataset..."
python3 create_fake_dataset.py

echo -e "\n[DRY RUN STEP 2/4] Running lightweight model training..."
python3 -m src.models.train_lightweight --config $CONFIG_FILE

echo -e "\n[DRY RUN STEP 3/4] Running ensemble model training..."
python3 -m src.models.train_ensembles --config $CONFIG_FILE

echo -e "\n[DRY RUN STEP 4/4] Running final evaluation..."
python3 -m src.models.evaluate_models --config $CONFIG_FILE

echo -e "\n=================================================="
echo "          PIPELINE DRY RUN COMPLETED SUCCESSFULLY!"
echo "=================================================="