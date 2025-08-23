import cv2
import json
import numpy as np
import os
import glob
from pathlib import Path

print("--- Starting Ground Truth Visualization Script ---")

# --- 1. SETUP: Configure your paths ---
# Path to the FOLDER containing your original test images



TEST_IMAGES_DIR = 'data/raw/Dental_dataset/test/img/'

# Folder where the prediction images will be saved

# Path to the FOLDER containing your original test annotations (JSON files)
TEST_ANNOTATIONS_DIR = 'data/raw/Dental_dataset/test/ann/'

OUTPUT_DIR = 'runs/ground_truth_visualizations/'

# --- 2. SCRIPT LOGIC ---
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
print(f"Annotated images will be saved in: {OUTPUT_DIR}")

# Get a list of all images to process
image_paths = glob.glob(os.path.join(TEST_IMAGES_DIR, '*.jpg')) # Use '*.png' if your files are png
print(f"Found {len(image_paths)} images to process.")

# Loop through each image to create an annotated version
for i, image_path in enumerate(image_paths):
    base_filename = os.path.basename(image_path)
    # Assumes JSON name is 'image.jpg.json'
    json_filename = f"{base_filename}.json"
    annotation_path = os.path.join(TEST_ANNOTATIONS_DIR, json_filename)
    output_path = os.path.join(OUTPUT_DIR, base_filename)

    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"({i+1}/{len(image_paths)}) SKIPPED Corrupted Image: {base_filename}")
        continue
    
    # Check if the annotation file for this image exists
    if not os.path.exists(annotation_path):
        print(f"({i+1}/{len(image_paths)}) SKIPPED No Annotation for: {base_filename}")
        continue
    
    # Read the JSON data and draw the annotations
    try:
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        
        for obj in data.get('objects', []):
            class_name = obj.get('classTitle', 'Unknown')
            points = obj.get('points', {}).get('exterior', [])
            if points:
                # Convert points to NumPy array for OpenCV
                pts = np.array(points, np.int32).reshape((-1, 1, 2))
                # Draw the polygon outline in GREEN
                cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                # Add the class name label
                cv2.putText(image, class_name, (points[0][0], points[0][1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Save the new image with the drawings
        cv2.imwrite(output_path, image)
        print(f"({i+1}/{len(image_paths)}) Saved visualization for '{base_filename}'")

    except Exception as e:
        print(f"An error occurred with {base_filename}: {e}")

print("\n----------------------------------")
print("✅ Visualization Complete!")
print(f"All annotated images have been saved to the '{OUTPUT_DIR}' folder.")
print("----------------------------------")