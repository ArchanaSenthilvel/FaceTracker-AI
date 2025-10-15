import cv2
import json
import os
import logging
from datetime import datetime
from pathlib import Path
import sys

from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from tracker import FaceTracker
from database import Database

# Configure logging
def setup_logging():
    """Setup logging configuration"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / 'events.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

class FaceTrackingSystem:
    def __init__(self, config_path='config.json'):
        """Initialize face tracking system"""
        logger.info("Initializing Face Tracking System...")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize components
        self.detector = FaceDetector(
            model_path=self.config.get('yolo_model', 'yolov8n-face.pt'),
            conf_threshold=self.config.get('detection_confidence', 0.5)
        )
        self.recognizer = FaceRecognizer(
            similarity_threshold=self.config.get('similarity_threshold', 0.6)
        )
        self.tracker = FaceTracker(
            max_disappeared=self.config.get('max_disappeared_frames', 30)
        )
        self.database = Database(
            connection_string=self.config.get('mongodb_uri', 'mongodb://localhost:27017/'),
            db_name=self.config.get('db_name', 'face_tracker')
        )
        
        # Load known embeddings
        self.known_embeddings = self.database.get_all_embeddings()
        
        # Setup directories
        self.setup_directories()
        
        # Tracking state
        self.frame_count = 0
        self.skip_frames = self.config.get('skip_frames', 2)
        self.tracked_entries = set()
        self.tracked_exits = set()
        self.next_face_id = len(self.known_embeddings) + 1
        
        logger.info("Face Tracking System initialized successfully")
    
    def setup_directories(self):
        """Create necessary directories"""
        today = datetime.now().strftime('%Y-%m-%d')
        self.entry_dir = Path(f'logs/entries/{today}')
        self.exit_dir = Path(f'logs/exits/{today}')
        self.entry_dir.mkdir(parents=True, exist_ok=True)
        self.exit_dir.mkdir(parents=True, exist_ok=True)
    
    def save_face_image(self, frame, bbox, face_id, event_type):
        """Save cropped face image"""
        try:
            x1, y1, x2, y2 = bbox
            face_img = frame[y1:y2, x1:x2]
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"{face_id}_{timestamp}.jpg"
            
            if event_type == 'entry':
                filepath = self.entry_dir / filename
            else:
                filepath = self.exit_dir / filename
            
            cv2.imwrite(str(filepath), face_img)
            logger.info(f"Saved {event_type} image: {filepath}")
            
            return str(filepath)
        except Exception as e:
            logger.error(f"Error saving face image: {e}")
            return None
    
    def process_frame(self, frame):
        """Process single frame"""
        self.frame_count += 1
        
        # Detect faces (skip frames as configured)
        if self.frame_count % (self.skip_frames + 1) == 0:
            detections = self.detector.detect_faces(frame)
        else:
            detections = []
        
        # Update tracker
        tracked_objects = self.tracker.update(detections)
        
        # Process tracked objects
        for obj_id, obj_data in tracked_objects.items():
            bbox = obj_data['bbox']
            face_id = obj_data['face_id']
            
            # If face not recognized yet and we have a detection
            if face_id is None and len(detections) > 0:
                embedding = self.recognizer.get_embedding(frame, bbox)
                
                if embedding is not None:
                    # Try to match with known faces
                    matched_id, similarity = self.recognizer.find_match(
                        embedding, 
                        self.known_embeddings
                    )
                    
                    if matched_id is not None:
                        # Known face
                        face_id = matched_id
                        logger.info(f"Recognized face: {face_id} (similarity: {similarity:.2f})")
                    else:
                        # New face - register
                        face_id = f"FACE_{self.next_face_id:04d}"
                        self.next_face_id += 1
                        
                        timestamp = datetime.now()
                        self.database.register_face(face_id, embedding, timestamp)
                        self.known_embeddings[face_id] = embedding
                        
                        logger.info(f"Registered new face: {face_id}")
                    
                    # Assign face ID to tracked object
                    self.tracker.assign_face_id(obj_id, face_id)
                    
                    # Log entry event (only once per face)
                    if face_id not in self.tracked_entries:
                        image_path = self.save_face_image(frame, bbox, face_id, 'entry')
                        self.database.log_event(
                            face_id, 
                            'entry', 
                            datetime.now(), 
                            image_path
                        )
                        self.tracked_entries.add(face_id)
                        logger.info(f"Face {face_id} entered frame")
        
        # Check for exits
        current_face_ids = set()
        for obj_data in tracked_objects.values():
            if obj_data['face_id']:
                current_face_ids.add(obj_data['face_id'])
        
        # Detect exits
        exited_faces = self.tracked_entries - current_face_ids - self.tracked_exits
        for face_id in exited_faces:
            # Find last known bbox for this face
            for obj_data in list(tracked_objects.values()):
                if obj_data['face_id'] == face_id:
                    bbox = obj_data['bbox']
                    image_path = self.save_face_image(frame, bbox, face_id, 'exit')
                    self.database.log_event(
                        face_id, 
                        'exit', 
                        datetime.now(), 
                        image_path
                    )
                    self.tracked_exits.add(face_id)
                    logger.info(f"Face {face_id} exited frame")
                    break
        
        return frame, tracked_objects
    
    def run(self, video_source):
        """Run face tracking on video source"""
        logger.info(f"Starting face tracking on: {video_source}")
        
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {video_source}")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video stream")
                    break
                
                processed_frame, tracked_objects = self.process_frame(frame)
                
                # Draw bounding boxes (optional visualization)
                for obj_id, obj_data in tracked_objects.items():
                    bbox = obj_data['bbox']
                    face_id = obj_data['face_id']
                    
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    if face_id:
                        cv2.putText(
                            processed_frame, 
                            face_id, 
                            (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (0, 255, 0), 
                            2
                        )
                
                # Display frame (optional)
                cv2.imshow('Face Tracking', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User interrupted processing")
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print statistics
            unique_count = self.database.get_unique_visitor_count()
            logger.info(f"Total unique visitors: {unique_count}")
            logger.info(f"Total entries logged: {len(self.tracked_entries)}")
            logger.info(f"Total exits logged: {len(self.tracked_exits)}")
            
            self.database.close()

if __name__ == '__main__':
    system = FaceTrackingSystem('config.json')
    
    # Use video file or RTSP stream
    video_source = 'sample_video.mp4'  # Change to RTSP URL for live stream
    
    system.run(video_source)