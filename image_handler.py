# image_handler.py

import os
import time

import cv2
import threading
from watchdog.events import FileSystemEventHandler
from datetime import datetime
from config import Config
from api_handler import save_attendance_to_api, update_client_via_api, create_client_via_api
from funcs import extract_date_from_filename

def process_image(file_path, camera_id, db_manager, face_processor, employee_last_report_times, client_last_report_times, lock):
    Config.logger.info(f"Processing image: {file_path} from camera_id: {camera_id}")
    try:
        image = cv2.imread(file_path)
        if image is None:
            Config.logger.error(f"Failed to read image from {file_path}")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, Config.DET_SIZE)

        embedding, age, gender = face_processor.get_embedding_from_image(image_resized)
        if embedding is None:
            Config.logger.error(f"No face embedding found in image: {file_path}")
            return

        # Set default age and gender if not detected
        age = int(round(age)) if age is not None else Config.DEFAULT_AGE
        gender = int(round(gender)) if gender is not None else Config.DEFAULT_GENDER

        timestamp = extract_date_from_filename(os.path.basename(file_path))
        if not timestamp:
            Config.logger.error(f"Could not extract date from filename: {file_path}")
            return

        # Search for matching employee
        employee, similarity_emp = db_manager.find_matching_employee(embedding)
        if employee:
            person_id = employee['person_id']
            with lock:
                last_report_time = employee_last_report_times.get(person_id)
                current_time = datetime.now()
                if last_report_time and (current_time - last_report_time).total_seconds() < Config.REPORT_COOLDOWN_SECONDS:
                    Config.logger.info(f"Employee {person_id} was seen recently. Skipping attendance report.")
                    return
                else:
                    save_attendance_to_api(
                        person_id=employee['person_id'],
                        device_id=camera_id,
                        image_path=file_path,
                        timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        score=similarity_emp
                    )
                    employee_last_report_times[person_id] = current_time
                    return

        # Search for matching client
        client, similarity_cli = db_manager.find_matching_client(embedding)
        if client:
            person_id = client['person_id']
            with lock:
                last_report_time = client_last_report_times.get(person_id)
                current_time = datetime.now()
                if last_report_time and (current_time - last_report_time).total_seconds() < Config.REPORT_COOLDOWN_SECONDS:
                    Config.logger.info(f"Client {person_id} was seen recently. Skipping visit history update.")
                    return
                else:
                    update_client_via_api(
                        client_id=person_id,
                        datetime_str=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        device_id=camera_id
                    )
                    client_last_report_times[person_id] = current_time
                    Config.logger.info(f"Client {person_id} visited with similarity {similarity_cli}")
            return

        # If no match found, create new client
        new_client_id = create_client_via_api(
            image_path=file_path,
            first_seen=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            last_seen=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            gender=gender,
            age=age
        )

        if new_client_id:
            # Store the embedding in MongoDB
            db_manager.add_client_embedding(new_client_id, embedding)
        else:
            Config.logger.error("Failed to create new client")

    except Exception as e:
        Config.logger.error(f"Error processing image {file_path}: {e}")
    finally:
        # Clean up the processed file
        if os.path.exists(file_path):
            os.remove(file_path)
        # Remove corresponding BACKGROUND file if it exists
        bg_file = file_path.replace('SNAP', 'BACKGROUND')
        if os.path.exists(bg_file):
            os.remove(bg_file)

# Image Handler for Watchdog
class ImageHandler(FileSystemEventHandler):
    def __init__(self, camera_id, db_manager, face_processor, employee_last_report_times, client_last_report_times, lock):
        self.camera_id = camera_id
        self.db_manager = db_manager
        self.face_processor = face_processor
        self.employee_last_report_times = employee_last_report_times
        self.client_last_report_times = client_last_report_times
        self.lock = lock

    def on_created(self, event):
        if event.is_directory:
            return
        filename = os.path.basename(event.src_path)
        if filename.endswith('SNAP.jpg'):
            Config.logger.info(f"New image detected: {event.src_path}")
            # Start a new thread to wait and process the image
            threading.Thread(
                target=self.wait_and_process_image,
                args=(event.src_path,),
                daemon=True
            ).start()

    def wait_and_process_image(self, file_path):
        max_attempts = 10
        wait_interval = 0.5  # seconds
        attempts = 0

        while attempts < max_attempts:
            if self.is_file_fully_written(file_path):
                Config.logger.info(f"File {file_path} is fully written. Proceeding to process.")
                # Now process the image
                process_image(
                    file_path,
                    self.camera_id,
                    self.db_manager,
                    self.face_processor,
                    self.employee_last_report_times,
                    self.client_last_report_times,
                    self.lock
                )
                return
            else:
                attempts += 1
                time.sleep(wait_interval)

        Config.logger.error(f"File {file_path} is not fully written after {max_attempts * wait_interval} seconds.")
        # Optionally, you can decide to process the file anyway or skip it

    def is_file_fully_written(self, file_path):
        try:
            # Check if file size remains the same over an interval
            initial_size = os.path.getsize(file_path)
            time.sleep(0.5)
            new_size = os.path.getsize(file_path)
            if initial_size == new_size and initial_size > 0:
                return True
            else:
                return False
        except Exception as e:
            Config.logger.error(f"Error checking if file is fully written: {e}")
            return False