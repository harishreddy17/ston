import albumentations as A
import pandas as pd
import cv2
import numpy as np
import os

# Load the CSV data
data = pd.read_csv('bucket_1_annots.csv')

# Normalize coordinates to fit 256x256 image dimensions
def normalize_coordinates(x, y, img_size=256):
    x_normalized = (x + img_size / 2).clip(0, img_size - 1).astype(int)
    y_normalized = (-y + img_size / 2).clip(0, img_size - 1).astype(int)
    return x_normalized, y_normalized

data['x_normalized'], data['y_normalized'] = normalize_coordinates(data['x'], data['y'])

# Augmentation pipeline using albumentations
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0, rotate_limit=0, p=0.0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
])

# Function to generate augmented images and masks for semantic segmentation
def generate_augmented_images_and_masks(data, img_size=256, start_mask_value=14):
    grouped = data.groupby(['image_id', 'threed_model'])
    line_pairs = [
        (1, 2), (1, 3), (1, 4), (4, 8), (4, 6), (4, 9),
        (7, 8), (5, 6), (9, 10), (10, 11), (11, 12), (12, 13)
    ]
    
    for image_id, group in grouped:
        # Create a blank black image and mask
        black_image = np.zeros((img_size, img_size, 3), dtype=np.uint8)  # Black background
        mask = np.zeros((img_size, img_size), dtype=np.uint8)  # Black mask

        # 1. Assign point masks (1 to 13)
        for point_id in range(1, 14):  # Point IDs from 1 to 13
            point = group[group['id'] == point_id][['x_normalized', 'y_normalized']]
            if not point.empty:
                x, y = point.iloc[0]['x_normalized'], point.iloc[0]['y_normalized']
                print(f"Assigning point mask {point_id} at ({x}, {y})")
                x, y = np.clip(x, 0, img_size-1), np.clip(y, 0, img_size-1)
                mask[y, x] = point_id  # Assign point mask from 1 to 13

        # 2. Assign line masks (14 to 25)
        for idx, (id1, id2) in enumerate(line_pairs):
            point1 = group[group['id'] == id1][['x_normalized', 'y_normalized']]
            point2 = group[group['id'] == id2][['x_normalized', 'y_normalized']]
            
            if not point1.empty and not point2.empty:
                # Convert normalized coordinates back to pixel coordinates
                x1, y1 = point1.iloc[0]['x_normalized'], point1.iloc[0]['y_normalized']
                x2, y2 = point2.iloc[0]['x_normalized'], point2.iloc[0]['y_normalized']
                
                # Ensure coordinates are within image bounds
                x1, y1 = np.clip(x1, 0, img_size-1), np.clip(y1, 0, img_size-1)
                x2, y2 = np.clip(x2, 0, img_size-1), np.clip(y2, 0, img_size-1)
                
                print(f"Assigning line mask {start_mask_value + idx} between points ({x1}, {y1}) and ({x2}, {y2})")
                mask[y1, x1] = start_mask_value + idx  # Assign line masks from 14 to 25
                mask[y2, x2] = start_mask_value + idx  # Assign line masks from 14 to 25

        # Check if the mask has all values from 0 to 25
        unique_values = np.unique(mask)
        print(f"Mask unique values after assignment: {unique_values}")

        # Apply augmentations
        augmented = augmentation_pipeline(image=black_image, mask=mask)
        augmented_image = augmented['image']
        augmented_mask = augmented['mask']

        # Save the augmented image and mask
        base_name = image_id[0].split('/')[1].split('.')[0]
        line_image_path = f"output_lines5/{base_name}.png"
        mask_image_path = f"output_masks5/{base_name}.png"
        
        os.makedirs(os.path.dirname(line_image_path), exist_ok=True)
        os.makedirs(os.path.dirname(mask_image_path), exist_ok=True)

        # Save the augmented image and mask
        cv2.imwrite(line_image_path, augmented_image)
        cv2.imwrite(mask_image_path, augmented_mask)

        print(f"Saving line image to: {line_image_path}")
        print(f"Saving mask image to: {mask_image_path}")

    print("Augmented images and masks saved in separate folders successfully.")

# Generate the augmented images and masks
generate_augmented_images_and_masks(data)
