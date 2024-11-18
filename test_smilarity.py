# test_similarity.py

import cv2
import numpy as np
from face_processor import FaceProcessor
from config import Config, setup_logger
import logging

# Define the compute_sim function
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

def compute_similarity_between_images(image_path1, image_path2):
    # Initialize FaceProcessor
    face_processor = FaceProcessor()
    # Load first image
    image1 = cv2.imread(image_path1)
    if image1 is None:
        print(f"Failed to read image from {image_path1}")
        return None
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image1_resized = cv2.resize(image1_rgb, Config.DET_SIZE)
    # Get embedding for first image
    embedding1, _, _ = face_processor.get_embedding_from_image(image1_resized)
    if embedding1 is None:
        print(f"No face embedding found in image: {image_path1}")
        return None
    # Load second image
    image2 = cv2.imread(image_path2)
    if image2 is None:
        print(f"Failed to read image from {image_path2}")
        return None
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image2_resized = cv2.resize(image2_rgb, Config.DET_SIZE)
    # Get embedding for second image
    embedding2, _, _ = face_processor.get_embedding_from_image(image2_resized)
    if embedding2 is None:
        print(f"No face embedding found in image: {image_path2}")
        return None
    # Compute similarity
    similarity = compute_sim(embedding1, embedding2)
    return similarity

# Example usage
if __name__ == '__main__':
    # Paths to your two images
    image_path1 = '/home/rocked/github.com/empol_time/images/diyarrr/camera_1_20241028145255723049_SNAP.jpg'
    image_path2 = '/home/rocked/github.com/empol_time/images/diyarrr/camera_1_20241028155343967258_SNAP.jpg'

    similarity = compute_similarity_between_images(image_path1, image_path2)
    if similarity is not None:
        print(f"Similarity between the two images: {similarity}")
    else:
        print("Could not compute similarity.")
