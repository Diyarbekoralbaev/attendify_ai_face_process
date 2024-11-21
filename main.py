# main.py
import os
import threading
import asyncio
import time
from config import Config
from database_manager import DatabaseManager
from face_processor import FaceProcessor
from image_handler import process_image, ImageHandler
from data_fetcher import fetch_and_store_data
from websocket_listener import websocket_listener
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from queue import Queue

# class ImageHandler(FileSystemEventHandler):
#     def __init__(self, camera_id, db_manager, face_processor, employee_last_report_times, client_last_report_times, lock):
#         self.camera_id = camera_id
#         self.db_manager = db_manager
#         self.face_processor = face_processor
#         self.employee_last_report_times = employee_last_report_times
#         self.client_last_report_times = client_last_report_times
#         self.lock = lock
#
#     def on_created(self, event):
#         if event.is_directory:
#             return
#         filename = os.path.basename(event.src_path)
#         if filename.endswith('SNAP.jpg'):
#             Config.logger.info(f"New image detected: {event.src_path}")
#             threading.Thread(
#                 target=process_image,
#                 args=(
#                     event.src_path,
#                     self.camera_id,
#                     self.db_manager,
#                     self.face_processor,
#                     self.employee_last_report_times,
#                     self.client_last_report_times,
#                     self.lock
#                 ),
#                 daemon=True
#             ).start()

class MainRunner:
    def __init__(self, images_folder):
        self.images_folder = images_folder
        self.db_manager = DatabaseManager()
        self.face_processor = FaceProcessor()
        self.logger = Config.logger
        self.employee_last_report_times = {}
        self.client_last_report_times = {}
        self.lock = threading.Lock()

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Initialize a queue for image processing tasks
        self.image_queue = Queue()

        # Start the worker thread
        self.worker_thread = threading.Thread(target=self.image_processing_worker, daemon=True)
        self.worker_thread.start()

    def run(self):
        self.logger.info(f"Starting image processing for: {self.images_folder}")

        test_camera_dir = os.path.join(self.images_folder, 'test_camera')
        os.makedirs(test_camera_dir, exist_ok=True)

        # Start the periodic fetch_and_store_data in a separate thread
        fetch_and_store_data(self.db_manager, self.face_processor)

        # Start the WebSocket listener in a separate thread
        ws_thread = threading.Thread(target=self.start_websocket_listener, daemon=True)
        self.logger.info("Starting WebSocket listener.")
        ws_thread.start()

        # Process existing images in the directory by adding them to the queue
        self.process_images_in_directory(test_camera_dir)

        # Now start the watchdog to monitor new images
        self.start_watchdog(test_camera_dir)

    def image_processing_worker(self):
        while True:
            try:
                # Get the next image path from the queue
                file_path = self.image_queue.get()
                if file_path is None:
                    # Sentinel value to stop the worker
                    break
                self.logger.info(f"Worker processing image: {file_path}")
                process_image(
                    file_path,
                    1,  # camera_id
                    self.db_manager,
                    self.face_processor,
                    self.employee_last_report_times,
                    self.client_last_report_times,
                    self.lock
                )
                self.image_queue.task_done()
            except Exception as e:
                self.logger.error(f"Error in image_processing_worker: {e}")

    def enqueue_image(self, file_path):
        self.image_queue.put(file_path)

    def process_images_in_directory(self, directory):
        # List all files ending with 'SNAP.jpg' in the directory
        for filename in os.listdir(directory):
            if filename.endswith('SNAP.jpg'):
                file_path = os.path.join(directory, filename)
                self.logger.info(f"Found image to process: {file_path}")
                self.enqueue_image(file_path)

    def start_watchdog(self, directory):
        event_handler = ImageHandler(
            camera_id=1,
            db_manager=self.db_manager,
            face_processor=self.face_processor,
            employee_last_report_times=self.employee_last_report_times,
            client_last_report_times=self.client_last_report_times,
            lock=self.lock,
            enqueue_image=self.enqueue_image  # Pass the enqueue function to the handler
        )
        observer = Observer()
        observer.schedule(event_handler, directory, recursive=False)
        observer.start()
        self.logger.info(f"Started watchdog observer on directory: {directory}")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

    def start_websocket_listener(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(websocket_listener(self.db_manager, self.face_processor))


if __name__ == '__main__':
    images_folder = Config.IMAGES_FOLDER
    runner = MainRunner(images_folder)
    runner.run()
