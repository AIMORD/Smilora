from ultralytics import YOLO

# UPDATE this path to the 'best.pt' file from your latest run
model = YOLO('/home/obamabinladin/Documents/AIMORD/Git_repos/Smilora/Dental_YOLO_Training/v5run_full_augmentation2/weights/best.pt') 

# Run validation on the 'test' split
metrics = model.val(split='test', data='dental_dataset.yaml')
print(metrics)