import cv2
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class FaceTracker:
    def __init__(self, max_disappeared=30):
        """Initialize face tracker"""
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared
        self.face_id_map = {}
        logger.info("Face tracker initialized")
    
    def register(self, centroid, bbox):
        """Register new tracked object"""
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'bbox': bbox,
            'face_id': None
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        return self.next_object_id - 1
    
    def deregister(self, object_id):
        """Deregister tracked object"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.face_id_map:
            del self.face_id_map[object_id]
    
    def update(self, detections):
        """Update tracked objects with new detections"""
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        input_centroids = []
        input_bboxes = []
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_centroids.append((cx, cy))
            input_bboxes.append(bbox)
        
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[oid]['centroid'] for oid in object_ids]
            
            from scipy.spatial import distance as dist
            D = dist.cdist(object_centroids, input_centroids)
            
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                if D[row, col] > 50:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id]['centroid'] = input_centroids[col]
                self.objects[object_id]['bbox'] = input_bboxes[col]
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
            
            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols
            
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            for col in unused_cols:
                self.register(input_centroids[col], input_bboxes[col])
        
        return self.objects
    
    def assign_face_id(self, object_id, face_id):
        """Assign recognized face ID to tracked object"""
        self.face_id_map[object_id] = face_id
        self.objects[object_id]['face_id'] = face_id
    
    def get_face_id(self, object_id):
        """Get face ID for tracked object"""
        return self.face_id_map.get(object_id, None)