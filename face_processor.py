# face_processor.py

# import torch
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import logging
from config import Config
from funcs import get_faces_data

class FaceProcessor:
    def __init__(self):
        # Initialize FaceAnalysis with desired models
        self.provider = 'CPUExecutionProvider'
        logging.info(f"Using provider: {self.provider}")
        self.app = FaceAnalysis(name='buffalo_l', providers=[self.provider])
        self.app.prepare(ctx_id=0)

    def get_embedding_from_image(self, image):
        faces = self.app.get(image)
        if not faces:
            return None, None, None
        # Get the face with the highest detection score
        face = get_faces_data(faces, min_confidence=Config.MIN_DETECTION_CONFIDENCE)
        if face:
            # Pose check
            if abs(face.pose[1]) > Config.POSE_THRESHOLD or abs(face.pose[0]) > Config.POSE_THRESHOLD:
                Config.logger.warning(f"Face pose exceeds threshold: pose={face.pose}")
                return None, None, None

            embedding = face.embedding
            # Normalize embedding
            norm = np.linalg.norm(embedding)
            Config.logger.debug(f"Embedding norm: {norm}")
            if norm == 0:
                Config.logger.warning("Detected face has zero norm embedding.")
                return None, None, None
            embedding = embedding / norm
            Config.logger.debug(f"Normalized embedding: {embedding}")
            age = getattr(face, 'age', None)
            gender = getattr(face, 'gender', None)
            return embedding, age, gender
        return None, None, None
