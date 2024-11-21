# funcs.py

import logging
import os
import re
import numpy as np
from datetime import datetime
import requests
import cv2
from config import Config

def extract_date_from_filename(filename):
    """Extract date from filename."""
    try:
        # Adjust the split index and format based on your filename pattern
        date_str = filename.split("_")[2]
        return datetime.strptime(date_str, "%Y%m%d%H%M%S%f")
    except Exception as e:
        Config.logger.error(f"Error extracting date from filename: {e}")
        return None

def get_faces_data(faces, min_confidence=0.6):
    """Select the face with the highest detection score above the minimum confidence."""
    if not faces:
        return None
    faces = [face for face in faces if face.det_score >= min_confidence]
    if not faces:
        return None
    # Return the face with the highest detection score
    return max(faces, key=lambda face: face.det_score)

def get_embedding_from_url(image_url, face_processor):
    try:
        headers = {'Authorization': f'Bearer {Config.API_TOKEN}'}
        response = requests.get(image_url, headers=headers)
        response.raise_for_status()
        image_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            Config.logger.error(f"Failed to decode image from URL: {image_url}")
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image_resized = cv2.resize(image_rgb, Config.DET_SIZE)
        embedding, age, gender = face_processor.get_embedding_from_image(image_rgb)
        if embedding is None:
            Config.logger.warning(f"No faces detected or pose exceeds threshold in image from URL: {image_url}")
            return None
        return embedding
    except Exception as e:
        Config.logger.error(f"Error fetching or processing image from URL {image_url}: {e}")
        return None

def compute_sim(feat1, feat2, logger=Config.logger):
    try:
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        # Normalize embeddings
        feat1_norm = np.linalg.norm(feat1)
        feat2_norm = np.linalg.norm(feat2)
        if feat1_norm == 0 or feat2_norm == 0:
            logger.error("One of the embeddings has zero norm.")
            return None
        feat1 = feat1 / feat1_norm
        feat2 = feat2 / feat2_norm
        sim = np.dot(feat1, feat2)
        return sim
    except Exception as e:
        logger.error(e)
        return None