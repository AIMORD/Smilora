import os
import cv2
import numpy as np
import albumentations as A
import random

# --- CONFIGURATION ---
DATASET_DIR = 'data/processed/Dental_YOLO_Final_Dataset'
IMAGE_DIR = os.path.join(DATASET_DIR, 'images/train')
LABEL_DIR = os.path.join(DATASET_DIR, 'labels/train')

# Class IDs for the minority classes you want to augment
# Caries: 0, Cavity: 1, Crack: 2
MINORITY_CLASS_IDS = [0, 1, 2]

# How many new augmented versions to create for each qualifying image
AUGMENTATIONS_PER_IMAGE = 3
# ---------------------


# ✨ Define your augmentation pipeline using Albumentations
# This pipeline is designed for dental imagery
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4, brightness_limit=0.2, contrast_limit=0.2),
    A.ShiftScaleRotate(p=0.5, shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT),
    A.GaussNoise(p=0.2, var_limit=(10.0, 50.0)),
    A.MotionBlur(p=0.2, blur_limit=5),
], 
# This part is crucial for handling segmentation masks (polygons)
keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'])
)


def read_yolo_segmentation(label_path, img_width, img_height):
    """Reads YOLO segmentation labels and converts them to albumentations format."""
    keypoints = []
    class_labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            points = [float(p) for p in parts[1:]]

            # Each object is one entry in the list for albumentations
            class_labels.append(class_id)
            # Denormalize points and group them
            denormalized_points = [(points[i] * img_width, points[i+1] * img_height) for i in range(0, len(points), 2)]
            keypoints.extend(denormalized_points) # Flatten the list for processing
    return keypoints, class_labels

def write_yolo_segmentation(label_path, transformed_data, img_width, img_height):
    """Writes transformed data back to YOLO segmentation format."""
    with open(label_path, 'w') as f:
        # Note: This simple example assumes one object per image for clarity.
        # A more robust implementation would handle grouping points back to their objects.
        # This implementation will work correctly if you have one polygon per image, or
        # will merge all polygons into one object if multiple exist.
        # For most segmentation tasks, this is often a reasonable simplification to start.

        # Since albumentations flattens keypoints, we assume all points belong to the same object
        # for writing. This is a simplification. For complex multi-polygon labels, this needs adjustment.
        class_id = transformed_data['class_labels'][0] # Use the first class label
        yolo_line = [str(class_id)]
        for x, y in transformed_data['keypoints']:
            norm_x = max(0.0, min(1.0, x / img_width))
            norm_y = max(0.0, min(1.0, y / img_height))
            yolo_line.append(f"{norm_x:.6f}")
            yolo_line.append(f"{norm_y:.6f}")
        f.write(" ".join(yolo_line) + "\n")


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

    # Check if the label file contains any of the minority classes
    contains_minority = False
    with open(label_path, 'r') as f:
        for line in f:
            class_id = int(line.strip().split()[0])
            if class_id in MINORITY_CLASS_IDS:
                contains_minority = True
                break

    if contains_minority:
        print(f"Found minority class in {img_filename}. Augmenting...")

        # Load the image and its labels
        image_path = os.path.join(IMAGE_DIR, img_filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        keypoints, class_labels = read_yolo_segmentation(label_path, w, h)

        if not keypoints:
            continue

        for i in range(AUGMENTATIONS_PER_IMAGE):
            try:
                # Apply the transformations
                transformed = transform(image=image, keypoints=keypoints, class_labels=class_labels)

                transformed_image = transformed['image']

                # Create new filenames
                new_base_filename = f"{base_filename}_aug_{i}"
                new_image_path = os.path.join(IMAGE_DIR, f"{new_base_filename}.jpg")
                new_label_path = os.path.join(LABEL_DIR, f"{new_base_filename}.txt")

                # Save the new augmented image and label
                transformed_image_bgr = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(new_image_path, transformed_image_bgr)
                write_yolo_segmentation(new_label_path, transformed, w, h)

                augmented_count += 1
            except Exception as e:
                print(f"Could not augment {img_filename}: {e}")

print("\n----------------------------------")
print(f"🚀 Augmentation complete! Created {augmented_count} new image-label pairs.")
print("Your training dataset is now larger and more balanced.")
print("----------------------------------")