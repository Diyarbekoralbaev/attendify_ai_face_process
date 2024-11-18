# data_fetcher.py

import requests
from config import Config
from funcs import get_embedding_from_url

def fetch_and_store_data(db_manager, face_processor):
    Config.logger.info("Starting fetch_and_store_data task")

    try:
        headers = {'Authorization': f'Bearer {Config.API_TOKEN}'}
        # Fetch Employees
        employees_response = requests.get(f"{Config.API_BASE_URL}/employee/employees", headers=headers)
        employees_response.raise_for_status()
        employees = employees_response.json()

        fetched_employee_ids = [emp['id'] for emp in employees]

        # Process and store employee embeddings
        for employee in employees:
            image_url = f"{Config.API_BASE_URL}/{employee['image']}"
            embedding = get_embedding_from_url(image_url, face_processor)
            if embedding is not None:
                db_manager.add_employee_embedding(employee['id'], embedding)
            else:
                Config.logger.error(f"Failed to get embedding for Employee ID: {employee['id']}")

        # Identify and remove deleted employees from MongoDB
        db_manager.remove_deleted_employees(fetched_employee_ids)

        # Fetch Clients
        clients_response = requests.get(f"{Config.API_BASE_URL}/client/clients", headers=headers)
        clients_response.raise_for_status()
        clients = clients_response.json()

        fetched_client_ids = [cli['id'] for cli in clients]

        # Process and store client embeddings
        for client in clients:
            image_url = f"{Config.API_BASE_URL}/{client['image']}"
            embedding = get_embedding_from_url(image_url, face_processor)
            if embedding is not None:
                db_manager.add_client_embedding(client['id'], embedding)
                Config.logger.info(f"Stored/Updated embedding for Client ID: {client['id']}")
            else:
                Config.logger.error(f"Failed to get embedding for Client ID: {client['id']}")

        # Identify and remove deleted clients from MongoDB
        db_manager.remove_deleted_clients(fetched_client_ids)

        Config.logger.info("fetch_and_store_data task completed successfully.")
    except Exception as e:
        Config.logger.error(f"Error in fetch_and_store_data: {e}")

