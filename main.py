# main.py

import os
import threading
import asyncio
import time
from config import Config
from database_manager import DatabaseManager
from face_processor import FaceProcessor
from image_handler import ImageHandler
from data_fetcher import fetch_and_store_data
from watchdog.observers import Observer
from websocket_listener import websocket_listener

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

    def run(self):
        self.logger.info(f"Starting directory observer for: {self.images_folder}")
        event_handler = ImageHandler(
            self.db_manager,
            self.face_processor,
            self.logger,
            self.employee_last_report_times,
            self.client_last_report_times,
            self.lock
        )
        observer = Observer()
        test_camera_dir = os.path.join(self.images_folder, 'test_camera')
        os.makedirs(test_camera_dir, exist_ok=True)
        observer.schedule(event_handler, path=test_camera_dir, recursive=False)
        observer.start()

        # Start the periodic fetch_and_store_data in a separate thread
        fetch_thread = threading.Thread(target=fetch_and_store_data, args=(self.db_manager, self.face_processor), daemon=True)
        fetch_thread.start()

        # Start the WebSocket listener in a separate thread
        ws_thread = threading.Thread(target=self.start_websocket_listener, daemon=True)
        self.logger.info("Starting WebSocket listener.")
        ws_thread.start()

        try:
            while True:
                time.sleep(1)  # Keep the main thread alive
        except KeyboardInterrupt:
            self.logger.info("Stopping directory observer.")
            observer.stop()
        observer.join()

    def start_websocket_listener(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(websocket_listener(self.db_manager, self.face_processor))

# Entry Point
if __name__ == '__main__':
    images_folder = Config.IMAGES_FOLDER
    runner = MainRunner(images_folder)
    runner.run()
