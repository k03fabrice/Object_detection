import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

class Detector():
    def __init__(self, filepath, confidence_threshold):
        self.filepath = filepath
        self.model_path = "models/yolov8n.pt"
        self.confidence_threshold = confidence_threshold

    def foward(self):
        model = YOLO(self.model_path)
        image = cv2.imread(self.filepath)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(image_rgb)[0]
        annotated_image = image_rgb.copy()
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
        boxes = results.boxes
        return boxes, results.names, annotated_image, colors

    def plot(self):
        original_image = cv2.imread(self.filepath)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        boxes, class_names, annotated_image, colors = self.foward()
        class_labels = {}
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            if confidence > self.confidence_threshold:
                # get class id and name
                class_id = int(box.cls[0])
                class_name = class_names[class_id]

                # get color for this class
                color = colors[class_id % len(colors)].tolist()

                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                class_labels[class_name] = color

        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.title('original')
        plt.imshow(original_image)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title('detected')
        plt.imshow(annotated_image)
        plt.axis('off')
        legend_handles = []
        for class_name, color in class_labels.items():
            normalized_color = np.array(color) / 255.0  # Normalize the color
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=class_name,
                                             markerfacecolor=normalized_color, markersize=10))
        plt.legend(handles=legend_handles, loc='upper right', title='Classes')
        plt.tight_layout()
        plt.show()