# Visual Object Detection System - Phone Finder

## Overview
This project involves the implementation of a prototype for a visual object detection system. The primary objective is to locate a phone dropped on the floor using a single RGB camera image. The solution utilizes a YOLO (You Only Look Once) model for object detection.

## Approach
Since the training labels provided contained only the normalized center coordinates, which are not ideal for YOLO, this presented a more fun and challenging opportunity for model training. Instead of resorting to commonly used online labeling tools like CVAT or Roboflow, I chose to work with classical computer vision techniques for creating the labels for YOLO. Utilizing OpenCV APIs, I identified the regions of interest using normalized center coordinates and generated bounding boxes around the phone (object) to create labels in the YOLO format. These labels were then fed into the training script for the YOLO model in train_phone_finder.py. Subsequently, find_phone.py utilized these trained weights to accurately predict the bounding box of the phone, showcasing a blend of classical and modern object detection methods.

## Scripts

### 1. train_phone_finder.py
This script is responsible for training the phone detection model. It takes a single command line argument: the path to a folder containing images and a `labels.txt` file `<path_to_find_phone>`.

#### Key Functions:
- `main`: Orchestrates the process of reading labels, processing images, splitting datasets, and training the model.
- `plot_rectangle_mask`: Applies a mask over the region of interest in the image.
- `get_yolo_format_coordinates`: Converts bounding box coordinates to YOLO format.
- `move_files`: Moves files between directories during dataset preparation.

#### Usage
`python train_phone_finder.py <path_to_find_phone>
`

### 2. find_phone.py
This script detects a phone in a given image using the trained model. It accepts a single command line argument: the path to the JPG image to be tested `<path_to_test_image>`.

#### Key Functions:
- `detect_phone`: Detects the phone in the given image and returns its normalized coordinates.

#### Usage
```
python find_phone.py <path_to_test_image>
```

## Requirements before starting any tests
- `pip install ultralytics`
- `pip install opencv-python`
- `pip install -U scikit-learn scipy matplotlib`
- `pip install numpy`

#### Usage
```
pip install -r requirements.txt
```