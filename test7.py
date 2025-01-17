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
    A.HorizontalFlip(p=0.0),
    A.VerticalFlip(p=0.0),
    A.RandomRotate90(p=0.0),
    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0, rotate_limit=0, p=0.0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
])

# Function to draw a grid on the image
def draw_grid(image, grid_size=32, color=(255, 255, 255), thickness=1):
    img_height, img_width = image.shape[:2]
    
    # Draw horizontal lines
    for y in range(0, img_height, grid_size):
        cv2.line(image, (0, y), (img_width, y), color, thickness)
    
    # Draw vertical lines
    for x in range(0, img_width, grid_size):
        cv2.line(image, (x, 0), (x, img_height), color, thickness)
    
    return image

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

        # Draw rectangles for each body part (region) on the black image and mask
        for idx, (id1, id2) in enumerate(line_pairs):
            point1 = group[group['id'] == id1][['x_normalized', 'y_normalized']]
            point2 = group[group['id'] == id2][['x_normalized', 'y_normalized']]

            if not point1.empty and not point2.empty:
        # Extract coordinates
                x1, y1 = point1.iloc[0]['x_normalized'], point1.iloc[0]['y_normalized']
                x2, y2 = point2.iloc[0]['x_normalized'], point2.iloc[0]['y_normalized']

        # Ensure coordinates are within image bounds
                x1, y1 = np.clip(x1, 0, img_size-1), np.clip(y1, 0, img_size-1)
                x2, y2 = np.clip(x2, 0, img_size-1), np.clip(y2, 0, img_size-1)

        # Assign mask labels for points
                mask[y1, x1] = id1  # Mask label for point 1
                mask[y2, x2] = id2  # Mask label for point 2

        # Draw points on the black image
                cv2.circle(black_image, (x1, y1), 2, (0, 255, 0), -1)  # Green point
                cv2.circle(black_image, (x2, y2), 2, (255, 0, 0), -1)  # Red point

        # Assign mask label for the line
                line_label = 14 + idx  # Line labels from 14–25
                cv2.line(black_image, (x1, y1), (x2, y2), (255, 255, 255), 1)
        
        # Draw the line label on the midpoint
                mid_point = ((x1 + x2) // 2, (y1 + y2) // 2)
                mask[mid_point[1], mid_point[0]] = line_label  # Label the midpoint of the line

        # Optional: Annotate the image with the line label
                cv2.putText(black_image, str(line_label), mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        # Apply augmentations
        augmented = augmentation_pipeline(image=black_image, mask=mask)
        augmented_image = augmented['image']
        augmented_mask = augmented['mask']

        # Debugging: Check the augmented mask before and after normalization
        print(f"Original Mask Values: {np.unique(mask)}")

        # Draw grid on the augmented image
        # augmented_image_with_grid = draw_grid(augmented_image.copy(), grid_size=32)

        # # Normalize the mask to ensure visibility
        # augmented_mask_normalized = cv2.normalize(augmented_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # print(f"Normalized Mask Values (after normalization): {np.unique(augmented_mask_normalized)}")

        # Define the output paths for lines and masks
        base_name = image_id[0].split('/')[1].split('.')[0]
        line_image_path = f"output_lines5/{base_name}.png"
        mask_image_path = f"output_masks5/{base_name}.png"
        
        os.makedirs(os.path.dirname(line_image_path), exist_ok=True)
        os.makedirs(os.path.dirname(mask_image_path), exist_ok=True)

        # Save the augmented image and mask
        cv2.imwrite(line_image_path, augmented_image)
        cv2.imwrite(mask_image_path, augmented_mask)

        # Debugging: Print the file save locations
        print(f"Saving line image to: {line_image_path}")
        print(f"Saving mask image to: {mask_image_path}")

    print("Augmented images and masks saved in separate folders successfully.")

# Generate the augmented images and masks
generate_augmented_images_and_masks(data)
