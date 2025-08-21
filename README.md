# Dental Disease Segmentation Project

This project uses YOLOv8 to train an instance segmentation model for detecting dental diseases like Caries, Cavity, Crack, and Tooth from images.

---
## Setup

1.  **Clone Repository:**
    ```bash
    git clone <your-repo-url>
    cd Dental_ML_Project
    ```

2.  **(Recommended)** Create and activate a Python virtual environment.

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
---
## Usage

1.  **Place Data:** Download the raw dataset (`Dental-images-dataset` folder) and place it inside the `data/raw/` directory.

2.  **Run Preprocessing:** From the project's root directory, run:
    ```bash
    python scripts/01_preprocess.py
    ```
    This will create the processed dataset in `data/processed/` and the `dental_dataset.yaml` config file.

3.  **Run Training:**
    ```bash
    python scripts/02_train.py
    ```
    This will start the training. Results, including the final trained models, will be saved in a new `runs/` folder.