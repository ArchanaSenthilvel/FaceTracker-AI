import numpy as np
import cv2
from insightface.app import FaceAnalysis
import logging

logger = logging.getLogger(__name__)

class FaceRecognizer:
    def __init__(self, similarity_threshold=0.6):
        """Initialize InsightFace for face recognition"""
        try:
            self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.similarity_threshold = similarity_threshold
            logger.info("Face recognizer initialized with InsightFace")
        except Exception as e:
            logger.error(f"Error initializing face recognizer: {e}")
            raise
    
    def get_embedding(self, frame, bbox):
        """Extract face embedding from bounding box"""
        try:
            x1, y1, x2, y2 = bbox
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size == 0:
                return None
            
            faces = self.app.get(face_img)
            if len(faces) > 0:
                return faces[0].embedding
            return None
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None
    
    def compare_embeddings(self, emb1, emb2):
        """Calculate cosine similarity between embeddings"""
        if emb1 is None or emb2 is None:
            return 0.0
        
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def find_match(self, embedding, known_embeddings):
        """Find matching face in known embeddings"""
        if embedding is None:
            return None, 0.0
        
        best_match_id = None
        best_similarity = 0.0
        
        for face_id, known_emb in known_embeddings.items():
            similarity = self.compare_embeddings(embedding, known_emb)
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match_id = face_id
        
        return best_match_id, best_similarity