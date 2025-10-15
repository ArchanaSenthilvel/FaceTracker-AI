import cv2
import numpy as np
from ultralytics import YOLO
import logging
import os

logger = logging.getLogger(__name__)

# Set environment variable to allow model loading
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = '0'

class FaceDetector:
    def __init__(self, model_path='yolov8n-face.pt', conf_threshold=0.5):
        """Initialize YOLO face detector"""
        try:
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
            logger.info(f"Face detector initialized with model: {model_path}")
        except Exception as e:
            logger.error(f"Error initializing face detector: {e}")
            raise
    
    def detect_faces(self, frame):
        """Detect faces in frame and return bounding boxes"""
        try:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            faces = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    faces.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf
                    })
            
            return faces
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []