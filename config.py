# config.py

import os
from dotenv import load_dotenv
import logging

load_dotenv()

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup a logger with a given name and log file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add them to handlers
    c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

class Config:
    CHECK_NEW_CLIENT = float(os.getenv('CHECK_NEW_CLIENT', 0.65))  # Similarity threshold for clients
    EMPLOYEE_SIMILARITY_THRESHOLD = float(os.getenv('EMPLOYEE_SIMILARITY_THRESHOLD', 0.65))  # Similarity threshold for employees
    MIN_DETECTION_CONFIDENCE = float(os.getenv('MIN_DETECTION_CONFIDENCE', 0.6))  # Minimum face detection confidence
    logger = setup_logger('MainRunner', 'logs/main.log')
    DIMENSIONS = int(os.getenv('DIMENSIONS', 512))
    DET_SIZE = tuple(map(int, os.getenv('DET_SIZE', '640,640').split(',')))
    API_BASE_URL = os.getenv('API_BASE_URL', 'http://10.30.10.136:8000')
    API_TOKEN = os.getenv('API_TOKEN', 'your_api_token_here')  # Ensure this is set in your .env
    IMAGES_FOLDER = os.getenv('IMAGES_FOLDER', '/path/to/images')  # Update with your images folder path

    DEFAULT_AGE = int(os.getenv('DEFAULT_AGE', 30))
    DEFAULT_GENDER = int(os.getenv('DEFAULT_GENDER', 0))  # 0 for female, 1 for male

    REPORT_COOLDOWN_SECONDS = int(os.getenv('REPORT_COOLDOWN_SECONDS', 60))  # Cooldown period for sending reports

    POSE_THRESHOLD = int(os.getenv('POSE_THRESHOLD', 30))  # Pose angle threshold
