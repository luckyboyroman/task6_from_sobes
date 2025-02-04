from ultralytics import YOLO
import cv2
import numpy as np


class CarDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def get_dominant_color(self, roi, k=3):
        roi = roi.reshape((-1, 3))
        roi = np.float32(roi)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(roi, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        unique, counts = np.unique(labels, return_counts=True)
        dominant_color = centers[np.argmax(counts)]

        return tuple(map(int, dominant_color))

    def detect_cars(self, image_path):
        image = cv2.imread(image_path)
        output_image = image.copy()
        dominant_color_l = []
        results = self.model(image)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = box.cls[0]
                class_name = self.model.names[int(class_id)]

                if class_name == 'car':
                    roi = output_image[y1:y2, x1:x2]

                    dominant_color = self.get_dominant_color(roi)
                    dominant_color_l.append(dominant_color)

                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    color_text = f'Color: {dominant_color}'
                    cv2.putText(output_image, color_text, (x1, y2 + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)

        return output_image, dominant_color_l