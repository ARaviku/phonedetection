import cv2
import numpy as np
import os
from pathlib import Path
import sys
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import shutil


def get_yolo_format_coordinates(box, image_width, image_height):

    width = max(box[:, 0]) - min(box[:, 0])
    height = max(box[:, 1]) - min(box[:, 1])

    # Calculate center of the box
    center_x = min(box[:, 0]) + width / 2
    center_y = min(box[:, 1]) + height / 2

    # Normalize by image size
    x_center_norm = center_x / image_width
    y_center_norm = center_y / image_height
    width_norm = width / image_width
    height_norm = height / image_height

    return 0, x_center_norm, y_center_norm, width_norm, height_norm

# Function to produce ROI - masked rectangle 
def plot_rectangle_mask(image, x, y, image_name, modified_labels_path):

    # Classical computer vision for image processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=255)
    height, width = image.shape[:2]
    cx, cy = int(x * width), int(y * height)
    top_left = (cx-20,cy+20) #approximation based on data
    bottom_right = (cx+20, cy-20)
    mask = np.zeros_like(gray)  
    cv2.rectangle(mask, top_left, bottom_right, 255, -1) 
    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
    contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    phone_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(phone_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    original_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(original_image, [box], 0, (0, 255, 0), 2)  
    yolo_coordinates = get_yolo_format_coordinates(box, width, height)
    
    # Save the YOLO formatted coordinates to a .txt file
    txt_output_path = os.path.join(modified_labels_path, f"{image_name.split('.')[0]}.txt")
    with open(txt_output_path, 'w') as f:
        f.write(' '.join(map(str, yolo_coordinates)) + '\n')
    return original_image

# move files
def move_files(file_list, source_dir, dest_dir, file_extension):
    for file_name in file_list:
        label_file_name = os.path.splitext(file_name)[0] + file_extension
        shutil.move(os.path.join(source_dir, label_file_name), os.path.join(dest_dir, label_file_name))


def main(dataset_path):
    if not os.path.exists(dataset_path):
        print(f"The provided dataset path does not exist: {dataset_path}")
        return
    
    # Define paths for images and labels
    images_path = os.path.join(dataset_path, 'images')
    modified_labels_path = os.path.join(dataset_path, 'modified_labels')

    labels_file = os.path.join(dataset_path, 'labels/labels.txt')
    
    # Creating directories for YOLO training and validation images and labels
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(modified_labels_path, exist_ok=True)
    

    with open(labels_file, 'r') as file:
        labels = file.readlines()

    for label in labels:
        parts = label.strip().split(' ')
        image_name, x, y = parts[0], float(parts[1]), float(parts[2])

        image_path = os.path.join(images_path, image_name)
        image = cv2.imread(image_path)

        if image is not None:
            masked_image = plot_rectangle_mask(image, x, y, image_name, modified_labels_path)

        else:
            print(f"Failed to load {image_name}")

    # Creating directories for YOLO training and validation images and labels
    train_images_dir = os.path.join(images_path, 'train')
    val_images_dir = os.path.join(images_path, 'val')
    train_labels_dir = os.path.join(dataset_path, 'labels', 'train')
    val_labels_dir = os.path.join(dataset_path, 'labels', 'val')

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)


    all_images = [f for f in os.listdir(images_path) if f.endswith(('.jpg'))]

    # Splitting images into training and validation sets
    train_images, val_images = train_test_split(all_images, test_size=0.2, random_state=42)

    # Moving image and label files to the corresponding train/val directories
    move_files(train_images, images_path, train_images_dir, '.jpg')
    move_files(val_images, images_path, val_images_dir, '.jpg')
    move_files(train_images, modified_labels_path, train_labels_dir, '.txt')
    move_files(val_images, modified_labels_path, val_labels_dir, '.txt')


    # Creating YOLOv8 training configuration file
    phonedetection_yaml_path = os.path.join(dataset_path, 'phonedetection.yaml')
    with open(phonedetection_yaml_path, 'w') as f:
        f.write(f"path: {dataset_path}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("\nnc: 1\n")
        f.write("\nname:\n  0: 'phone'\n")
 

    model = YOLO("yolov8n.yaml") 
    yaml_path = os.path.join(dataset_path, "phonedetection.yaml")
    print(yaml_path)
    results = model.train(data=yaml_path, epochs=500, imgsz=490, rect=True)

if __name__ == "__main__":

    dataset_path = sys.argv[1]  
    print(dataset_path)
    main(dataset_path)