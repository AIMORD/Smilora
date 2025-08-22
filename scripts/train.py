import torch
from ultralytics import YOLO

model = YOLO('yolov8s-seg.pt')

# Train the model with a comprehensive set of augmentations
results = model.train(
    data='dental_dataset.yaml',
    epochs=150,
    imgsz=640,
    project='Dental_YOLO_Training',
    name='run_full_augmentation', # A new name for this experiment
    
    # --- Advanced Augmentations ---
    mixup=0.1,         # Mix two images and their labels (10% probability)
    copy_paste=0.1,    # Copy objects and paste them on other images (10% probability)
    
    # --- Geometric Augmentations ---
    degrees=15,        # Random rotation (-15 to +15 degrees)
    translate=0.1,     # Randomly shift image horizontally and vertically by 10%
    scale=0.2,         # Randomly zoom in or out by 20%
    shear=5,           # Shear the image by +/- 5 degrees
    perspective=0.001, # Small probability of a random perspective warp
    fliplr=0.5,        # 50% chance of a horizontal flip
    
    # --- Color Augmentations ---
    hsv_h=0.015,       # Adjust hue by +/- 1.5%
    hsv_s=0.7,         # Adjust saturation by +/- 70%
    hsv_v=0.4          # Adjust brightness by +/- 40%
)