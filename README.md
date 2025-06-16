# A Multi-Faceted Deep Learning Framework for Enhanced Cervical Cancer Diagnosis

This repository contains the code and technical report for a binary classifier for Cervical Cancer Whole Slide Images (WSIs). Multiple strategies were tested, and our proposed model was developed to address the challenge.

Our approach consists of two distinct models. The primary model utilizes Ensemble-Stacking and Grad-CAM, achieving an accuracy of **~99% on the SIPaKMeD Dataset** and **~90% on the Herlev dataset**. A secondary, lightweight model was also developed for segmentation tasks, achieving 92% accuracy.

## Project Structure

```
pap_smear_project/
├── configs/              # Project configurations (config.yml, config_test.yml)
├── data/
│   └── raw/              # Raw datasets should be placed here
├── notebooks/            # Jupyter notebooks for experimentation
├── saved_models/         # Saved .keras models after training
├── scripts/              # Automation scripts (run_dry_run.sh, etc.)
├── src/                  # Main Python source code
├── tests/                # Project tests
├── Dockerfile            # Docker configuration for containerization
├── .dockerignore         # Specifies files to exclude from the Docker image
├── environment.yml       # Conda environment file
├── requirements.txt      # Pip requirements file
└── README.md
```

## Setup Instructions

### 1. Clone the Repository

### 2. Download Data
Place the **Herlev** and **Sipakmed** datasets inside the `data/raw/` directory. The expected structure is:
```
data/raw/
├── HerlevData/
│   ├── test/
│   └── train/
├── im_Abnormal/
├── im_Carcinoma-in-situ/
├── ... (and other Sipakmed folders)
└── im_Superficial-Intermediate/
```
The data path is configured in `configs/config.yml`.

### 3. Environment Setup

You can set up the environment using Docker (recommended for consistency) or locally with a virtual environment.

#### Option A: Using Docker (Recommended)
This is the most reliable method as it creates a self-contained, consistent environment.

**Prerequisites:**
*   [Docker](https://www.docker.com/get-started) installed and running.
*   For GPU acceleration, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

**Steps:**
1.  **Build the Docker image:**
    ```bash
    docker build -t pap-smear-project .
    ```

2.  **Run the container:** This command starts an interactive shell inside the container and mounts your project directory, so any code changes are immediately reflected.

    *   **With GPU support:**
        ```bash
        docker run -it --rm --gpus all -v $(pwd):/app pap-smear-project
        ```
    *   **CPU only:**
        ```bash
        docker run -it --rm -v $(pwd):/app pap-smear-project
        ```
You are now inside the container's shell and can proceed to the "How to Run the Pipeline" section.

#### Option B: Local Setup (venv)
1.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2.  **Install Dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    ```

3.  **Make the Project "Editable":** This crucial step makes your `src` code importable.
    ```bash
    pip install -e .
    ```

4.  **macOS Users Only: Install SSL Certificates:**
    If you are on a Mac, Python may require you to explicitly install SSL certificates to download model weights.
    *   Open **Finder** -> **Applications** -> `Python 3.12` (or your version) folder.
    *   Double-click on **`Install Certificates.command`**.

## How to Run the Pipeline

This project includes automation scripts to simplify the workflow.

### 1. Run a Quick Dry Run (Recommended First)
A dry run tests the entire pipeline on a minimal, auto-generated dataset. It's the best way to verify that your environment is set up correctly without needing the real data or a GPU. It should complete in a few minutes.

```bash
bash scripts/run_dry_run.sh
```
If this script finishes without errors, your setup is perfect.

*(Note: This script automatically calls `create_fake_dataset.py` to generate a temporary, minimal dataset for the test. This fake data is placed in `data/raw/` and can be safely deleted afterwards.)*

### 2. Run the Full Training Pipeline
This will run the entire process on the real dataset as configured in `configs/config.yml`.

**Warning:** This is computationally expensive and will take a significant amount of time, especially without a GPU. Make sure you have downloaded the real datasets into `data/raw/` before running this.

```bash
bash scripts/run_full_training.sh
```

## Running Tests
To run the static smoke tests, which quickly check for import errors and configuration issues, use `pytest`:
```bash
pytest
```
If all tests pass, your environment is likely configured correctly.
