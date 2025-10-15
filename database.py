from pymongo import MongoClient
from datetime import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, connection_string='mongodb://localhost:27017/', db_name='face_tracker'):
        """Initialize MongoDB connection"""
        try:
            self.client = MongoClient(connection_string)
            self.db = self.client[db_name]
            self.faces = self.db.faces
            self.events = self.db.events
            self.stats = self.db.stats
            logger.info(f"Connected to MongoDB database: {db_name}")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def register_face(self, face_id, embedding, timestamp):
        """Register a new face in database"""
        try:
            face_doc = {
                'face_id': face_id,
                'embedding': embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                'first_seen': timestamp,
                'last_seen': timestamp,
                'total_visits': 1
            }
            self.faces.insert_one(face_doc)
            logger.info(f"Registered new face: {face_id}")
            return True
        except Exception as e:
            logger.error(f"Error registering face: {e}")
            return False
    
    def log_event(self, face_id, event_type, timestamp, image_path):
        """Log entry/exit event"""
        try:
            event_doc = {
                'face_id': face_id,
                'event_type': event_type,
                'timestamp': timestamp,
                'image_path': image_path
            }
            self.events.insert_one(event_doc)
            logger.info(f"Logged {event_type} event for face: {face_id}")
            return True
        except Exception as e:
            logger.error(f"Error logging event: {e}")
            return False
    
    def update_face_stats(self, face_id, timestamp):
        """Update face statistics"""
        try:
            self.faces.update_one(
                {'face_id': face_id},
                {
                    '$set': {'last_seen': timestamp},
                    '$inc': {'total_visits': 1}
                }
            )
            return True
        except Exception as e:
            logger.error(f"Error updating face stats: {e}")
            return False
    
    def get_all_embeddings(self):
        """Retrieve all face embeddings"""
        try:
            embeddings = {}
            for face in self.faces.find():
                embeddings[face['face_id']] = np.array(face['embedding'])
            return embeddings
        except Exception as e:
            logger.error(f"Error retrieving embeddings: {e}")
            return {}
    
    def get_unique_visitor_count(self):
        """Get count of unique visitors"""
        try:
            return self.faces.count_documents({})
        except Exception as e:
            logger.error(f"Error getting visitor count: {e}")
            return 0
    
    def close(self):
        """Close database connection"""
        self.client.close()
        logger.info("Database connection closed")