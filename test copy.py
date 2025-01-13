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
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
])

# Function to generate augmented images and masks for semantic segmentation
def generate_augmented_images_and_masks(data, img_size=256, start_mask_value=1, max_mask_value=25):
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
                # Convert normalized coordinates back to pixel coordinates
                x1, y1 = point1.iloc[0]['x_normalized'], point1.iloc[0]['y_normalized']
                x2, y2 = point2.iloc[0]['x_normalized'], point2.iloc[0]['y_normalized']
                
                # Ensure coordinates are within image bounds
                x1, y1 = np.clip(x1, 0, img_size-1), np.clip(y1, 0, img_size-1)
                x2, y2 = np.clip(x2, 0, img_size-1), np.clip(y2, 0, img_size-1)

                # Debugging: print the coordinates
                print(f"Drawing line from ({x1}, {y1}) to ({x2}, {y2})")

                # Draw the points and line (debugging step)
                cv2.circle(black_image, (x1, y1), 2, (0, 255, 0), -1)  # Green point at x1, y1
                cv2.circle(black_image, (x2, y2), 2, (255, 0, 0), -1)  # Red point at x2, y2
                cv2.line(black_image, (x1, y1), (x2, y2), (255, 255, 255), 1)  # White line

                # Calculate the other two corners to form a rectangle
                rect_x1, rect_y1 = min(x1, x2), min(y1, y2)
                rect_x2, rect_y2 = max(x1, x2), max(y1, y2)

                # Debugging: print the rectangle coordinates
                print(f"Drawing rectangle from ({rect_x1}, {rect_y1}) to ({rect_x2}, {rect_y2})")

                # Draw rectangle on the mask
                cv2.rectangle(mask, (rect_x1, rect_y1), (rect_x2, rect_y2), start_mask_value + idx, -1)

        # Apply augmentations (same for both image and mask)
        augmented = augmentation_pipeline(image=black_image, mask=mask)
        augmented_image = augmented['image']
        augmented_mask = augmented['mask']

        # Normalize the mask to ensure visibility
        augmented_mask_normalized = cv2.normalize(augmented_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Debugging: Save the normalized mask and image
        cv2.imwrite("debug_augmented_image.png", augmented_image)
        cv2.imwrite("debug_augmented_mask.png", augmented_mask_normalized)

        # Visualize the mask overlayed on the image (for better visibility)
        mask_overlay = augmented_image.copy()

        # Apply yellow color overlay (BGR format: [0, 255, 255] for yellow)
        # Ensure we overlay the color only where the mask exists (where mask value > 0)
        mask_overlay[augmented_mask_normalized > 0] = [0, 255, 255]  # Yellow color for mask overlay

        # Apply the yellow overlay directly to the original augmented image
        augmented_image[augmented_mask_normalized > 0] = [0, 255, 255]  # Yellow color for mask overlay

        # Save the overlay image
        cv2.imwrite("debug_mask_overlay.png", mask_overlay)  # Debugging image with overlay
        
        # Save the final augmented image with mask applied
        base_name = image_id[0].split('/')[1].split('.')[0]
        output_image_path = f"output_augmented/{base_name}_lines_aug.png"
        output_mask_path = f"output_augmented/{base_name}_mask_aug.png"
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

        # Save the image with overlay (final output)
        cv2.imwrite(output_image_path, augmented_image)  # Final output with mask
        cv2.imwrite(output_mask_path, augmented_mask_normalized)  # Save the normalized mask

    print("Augmented images and masks generated successfully.")

# Generate the augmented images and masks
generate_augmented_images_and_masks(data)
