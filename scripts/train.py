import torch
from ultralytics import YOLO

# Load a pretrained model to start training
model = YOLO('yolov8n-seg.pt')

# Start the training and stop automatically at 20 epochs
results = model.train(
    data='dental_dataset.yaml',
    epochs=150,      # <-- Set to the target for Day 1
    imgsz=640,
    project='Dental_YOLO_Training',
    name='automated_run_100_epochs',# A consistent name for the project
    device=0

)