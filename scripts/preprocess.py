import os
import json
import shutil
import yaml

CLASS_MAP = {
    'Caries': 0,
    'Cavity': 1,
    'Crack': 2,
    'Tooth':3
}

DATASET_ROOT_PATH = 'data/raw/Dental_dataset'
OUTPUT_DIR = 'data/processed/Dental_YOLO_Final_Dataset'

for split in ['train', 'valid', 'test']:
    os.makedirs(os.path.join(OUTPUT_DIR, f'images/{split}'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, f'labels/{split}'), exist_ok=True)

def process_split(split_name):
  src_img_dir = os.path.join(DATASET_ROOT_PATH, split_name, 'img/')
  src_ann_dir = os.path.join(DATASET_ROOT_PATH, split_name, 'ann/')

  filenames = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
  total_files = len(filenames)

  for i, img_filename in enumerate(filenames):
    base_filename = os.path.splitext(img_filename)[0]
    json_filename = f"{img_filename}.json"

    src_img_path = os.path.join(src_img_dir, img_filename)
    src_ann_path = os.path.join(src_ann_dir, json_filename)

    if not os.path.exists(src_ann_path):
      print(f"\nWarning: Annotation for '{img_filename}' not found. Skipping this file.")
      continue


    with open(src_ann_path, 'r') as f:
      ann_data = json.load(f)

    img_width = ann_data['size']['width']
    img_height = ann_data['size']['height']

    yolo_labels = []

    for obj in ann_data['objects']:
      class_title = obj['classTitle']
      if class_title in CLASS_MAP:
        class_index = CLASS_MAP[class_title]
        points = obj['points']['exterior']

        normalized_points = []
        for x,y in points:
          norm_x = x / img_width
          norm_y = y / img_height
          normalized_points.extend([norm_x, norm_y])

        if normalized_points:
          yolo_line = f"{class_index} " + " ".join(f"{p:.6f}" for p in normalized_points)
          yolo_labels.append(yolo_line)


    if yolo_labels:
      dest_img_path = os.path.join(OUTPUT_DIR, f'images/{split_name}/{img_filename}')
      dest_label_path = os.path.join(OUTPUT_DIR, f'labels/{split_name}/{base_filename}.txt')
      with open(dest_label_path, 'w') as f:
        f.write("\n".join(yolo_labels))
      shutil.copy2(src_img_path, dest_img_path)

process_split('train')
process_split('valid')
process_split('test')

print("\n----------------------------------")
print("ALL PREPROCESSING IS COMPLETE!")
print(f"Your final, model-ready dataset is now in: {OUTPUT_DIR}")
print("----------------------------------")

yaml_data = {
    'train': os.path.abspath(os.path.join(OUTPUT_DIR, 'images/train')),
    'val': os.path.abspath(os.path.join(OUTPUT_DIR, 'images/valid')),
    'test': os.path.abspath(os.path.join(OUTPUT_DIR, 'images/test')),
    'nc': len(CLASS_MAP),
    'names': list(CLASS_MAP.keys())
}


with open('dental_dataset.yaml', 'w') as f:
    yaml.dump(yaml_data, f, default_flow_style=False)