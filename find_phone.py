import cv2
import sys
from ultralytics import YOLO
import os

def detect_phone(model_path, image_path):
    image = cv2.imread(image_path)
    model = YOLO(model_path)
    results = model.predict(image)

    for result in results:
        boxes = result.boxes
        if boxes.xywhn[0].shape[0]: 
            x_center_norm, y_center_norm = boxes.xywhn[0][0].item(), boxes.xywhn[0][1].item()
            return x_center_norm, y_center_norm
    return None

def main(image_path):
    # model_path = '/home/annu/Desktop/Project_codes/phonedetection_submission/runs/detect/train/weights/best.pt'
    model_path = os.path.join('runs', 'detect', 'train', 'weights', 'best.pt')
    coordinates = detect_phone(model_path, image_path)

    if coordinates:
        print(f"{coordinates[0]:.4f} {coordinates[1]:.4f}")
    else:
        print("No phone detected.")

if __name__ == "__main__":
    image_path = sys.argv[1]
    main(image_path)
