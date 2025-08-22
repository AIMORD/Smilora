import os
import cv2
import numpy as np
import albumentations as A
import random

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, 'data/processed/Dental_YOLO_Final_Dataset')

IMAGE_DIR = os.path.join(DATASET_DIR, 'images/train')
LABEL_DIR = os.path.join(DATASET_DIR, 'labels/train')

MINORITY_CLASS_IDS = [0, 1, 2]
AUGMENTATIONS_PER_IMAGE = 3
# ---------------------


# ✨ Define your augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4, brightness_limit=0.2, contrast_limit=0.2),
    A.ShiftScaleRotate(p=0.5, shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT),
    A.GaussNoise(p=0.2),
    A.MotionBlur(p=0.2, blur_limit=5),
],
keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'])
)


def read_yolo_segmentation(label_path, img_width, img_height):
    keypoints = []
    class_labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            points_normalized = [float(p) for p in parts[1:]]

            denormalized_points = []
            for i in range(0, len(points_normalized), 2):
                x = points_normalized[i] * img_width
                y = points_normalized[i+1] * img_height
                denormalized_points.append((x, y))

            keypoints.extend(denormalized_points)
            class_labels.extend([class_id] * len(denormalized_points))

    return keypoints, class_labels

def write_yolo_segmentation(label_path, transformed_data, img_width, img_height):
    objects = {}
    for kp, label in zip(transformed_data['keypoints'], transformed_data['class_labels']):
        if label not in objects:
            objects[label] = []
        norm_x = max(0.0, min(1.0, kp[0] / img_width))
        norm_y = max(0.0, min(1.0, kp[1] / img_height))
        objects[label].extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])

    with open(label_path, 'w') as f:
        for class_id, points in objects.items():
            line = f"{class_id} " + " ".join(points)
            f.write(line + "\n")


# --- MAIN SCRIPT LOGIC ---
print("Starting offline augmentation for minority classes...")
image_files = os.listdir(IMAGE_DIR)
augmented_count = 0

for img_filename in image_files:
    if not img_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    base_filename = os.path.splitext(img_filename)[0]
    label_filename = f"{base_filename}.txt"
    label_path = os.path.join(LABEL_DIR, label_filename)

    if not os.path.exists(label_path):
        continue

    contains_minority = False
    with open(label_path, 'r') as f:
        for line in f:
            class_id = int(line.strip().split()[0])
            if class_id in MINORITY_CLASS_IDS:
                contains_minority = True
                break

    if contains_minority:
        print(f"Found minority class in {img_filename}. Augmenting...")

        image_path = os.path.join(IMAGE_DIR, img_filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        keypoints, class_labels = read_yolo_segmentation(label_path, w, h)

        if not keypoints:
            print(f"Warning: No keypoints found for {img_filename}. Skipping.")
            continue

        for i in range(AUGMENTATIONS_PER_IMAGE):
            try:
                transformed = transform(image=image, keypoints=keypoints, class_labels=class_labels)
                transformed_image = transformed['image']
                
                new_base_filename = f"{base_filename}_aug_{i}"
                new_image_path = os.path.join(IMAGE_DIR, f"{new_base_filename}.jpg")
                new_label_path = os.path.join(LABEL_DIR, f"{new_base_filename}.txt")

                # --- ✅ FIX: Corrected the typo from COLOR_RGB_BGR to COLOR_RGB2BGR ---
                transformed_image_bgr = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(new_image_path, transformed_image_bgr)
                write_yolo_segmentation(new_label_path, transformed, w, h)

                augmented_count += 1
            except Exception as e:
                print(f"Could not augment {img_filename}: {e}")

print("\n----------------------------------")
print(f"🚀 Augmentation complete! Created {augmented_count} new image-label pairs.")
if augmented_count > 0:
    print("Your training dataset is now larger and more balanced.")
else:
    print("No new images were created. Please check for other errors if this count is zero.")
print("----------------------------------")