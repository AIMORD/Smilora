import torch
from ultralytics import YOLO

# Load a pretrained model to start a new training run
model = YOLO('yolov8n-seg.pt')

# Start the training with performance-enhancing changes
results = model.train(
    data='dental_dataset.yaml',
    epochs=20,
    imgsz=640,
    project='Dental_YOLO_Training',
    name='run_withou_aug', 

)