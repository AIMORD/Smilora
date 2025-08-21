import torch
from ultralytics import YOLO

# Load a pretrained model to start a new training run
model = YOLO('yolov8s-seg.pt')

# Start the training with performance-enhancing changes
results = model.train(
    data='dental_dataset.yaml',
    epochs=150,
    imgsz=640,
    project='Dental_YOLO_Training',
    device=0,
    name='V3(run_with_aug_yolo_small)', 
    cls=1.5,
    copy_paste=0.2
)