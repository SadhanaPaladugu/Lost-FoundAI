"""
Core object detection functionality using YOLO11.
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import streamlit as st
from ultralytics import YOLO
from config.settings import MODEL_PATH, CONFIDENCE_THRESHOLD

class ObjectDetector:
    """Handles object detection using YOLO11."""
    
    def __init__(self):
        self.model = self._load_model()

    @st.cache_resource
    def _load_model(_self):
        """Load YOLO model with caching. Note: _self instead of self"""
        try:
            return YOLO(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading YOLO11n model: {e}")
            st.info(f"Make sure {MODEL_PATH} is in the correct directory")
            return None

    def detect(self, image: np.ndarray, target_object: Optional[str] = None
              ) -> Tuple[np.ndarray, List[Dict]]:
        """Detect objects in an image."""
        if self.model is None:
            return image, []
        results = self.model(image, conf=CONFIDENCE_THRESHOLD)
        detections = []
        annotated_image = image.copy()
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    detections.append({
                        'class': class_name,
                        'confidence': float(confidence),
                        'bbox': (int(x1), int(y1), int(x2), int(y2))
                    })
                    color = (0, 255, 0) if (target_object and target_object.lower() in class_name.lower()) else (255, 165, 0)
                    thickness = 3 if (target_object and target_object.lower() in class_name.lower()) else 2
                    cv2.rectangle(
                        annotated_image,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color, thickness)
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(
                        annotated_image, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return annotated_image, detections
