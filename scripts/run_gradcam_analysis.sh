#!/bin/bash
# ==============================================================================
# run_gradcam_analysis.sh
#
# This script runs the Grad-CAM visualization pipeline.
#
# It uses the settings in the `gradcam_analysis` section of the config file
# to load a specific model and generate attention heatmaps on test images.
#
# Usage:
#   bash scripts/run_gradcam_analysis.sh
# ==============================================================================

set -e

echo "=================================================="
echo "          STARTING GRAD-CAM ANALYSIS"
echo "=================================================="

# Run the main Python script for Grad-CAM
python3 -m src.visualization.run_gradcam --config configs/config.yml

echo -e "\n=================================================="
echo "       GRAD-CAM ANALYSIS COMPLETED SUCCESSFULLY!"
echo "=================================================="