# api_handler.py

import os
import requests
from config import Config
from funcs import get_embedding_from_url

def save_attendance_to_api(person_id, device_id, image_path, timestamp, score):
    """Send attendance data to FastAPI API"""
    endpoint = "/attendance/create"  # Adjust as per actual API endpoint
    data = {
        'employee': person_id,
        'device_id': device_id,
        'datetime': timestamp,
        'score': score
    }
    try:
        headers = {'Authorization': f'Bearer {Config.API_TOKEN}'}
        with open(image_path, 'rb') as img_file:
            files = {
                'image': (os.path.basename(image_path), img_file, 'image/jpeg')
            }
            response = send_report(endpoint, data=data, files=files, headers=headers)
            if response:
                Config.logger.info(f"Attendance sent for employee {person_id} with similarity {score}")
    except Exception as e:
        Config.logger.error(f"Error sending attendance to API: {e}")

def update_client_via_api(client_id, datetime_str, device_id):
    """Send client visit data to FastAPI API"""
    endpoint = f"/clients/visit-history/"
    data = {
        'datetime': datetime_str,
        'device_id': device_id,
        'client': client_id
    }
    try:
        headers = {'Authorization': f'Bearer {Config.API_TOKEN}'}
        response = send_report_json(endpoint, data=data, headers=headers)
        if response:
            Config.logger.info(f"Client {client_id} visit updated.")
    except Exception as e:
        Config.logger.error(f"Error updating client visit via API: {e}")

def create_client_via_api(image_path, first_seen, last_seen, gender, age):
    """Create a new client via FastAPI API and return the new client ID"""
    gender_named = "unknown"
    if gender == 0:
        gender_named = "female"
    elif gender == 1:
        gender_named = "male"

    endpoint = "/clients"
    data = {
        'first_seen': first_seen,
        'last_seen': last_seen,
        'gender': gender_named,
        'age': age
    }
    try:
        headers = {'Authorization': f'Bearer {Config.API_TOKEN}'}
        with open(image_path, 'rb') as img_file:
            files = {
                'image': (os.path.basename(image_path), img_file, 'image/jpeg')
            }
            response = send_report_with_response(endpoint, data=data, files=files, headers=headers)
            if response and response.status_code == 200:
                client_data = response.json()
                new_client_id = client_data.get('data', {}).get('id')
                if new_client_id:
                    Config.logger.info(f"New client created with ID: {new_client_id}")
                    return new_client_id
                else:
                    Config.logger.error("New client ID not found in the API response.")
                    return None
            else:
                Config.logger.error(f"Failed to create new client. Status Code: {response.status_code if response else 'No Response'}")
                return None
    except Exception as e:
        Config.logger.error(f"Error creating new client via API: {e}")
        return None

def send_report(endpoint, data=None, files=None, headers=None):
    url = f"{Config.API_BASE_URL}{endpoint}"
    try:
        response = requests.post(url, data=data, files=files, headers=headers)
        response.raise_for_status()
        Config.logger.info(f"Successfully sent report to {endpoint}")
        return response
    except requests.RequestException as e:
        Config.logger.error(f"Failed to send report to {endpoint}: {e}")
        return None

def send_report_json(endpoint, data=None, headers=None):
    """Send JSON report to FastAPI API"""
    url = f"{Config.API_BASE_URL}{endpoint}"
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        Config.logger.info(f"Successfully sent JSON report to {endpoint}")
        return response
    except requests.RequestException as e:
        # Attempt to log the response content for detailed error information
        try:
            error_content = response.json()
            Config.logger.error(f"Failed to send JSON report to {endpoint}: {e}, Response: {error_content}")
        except Exception:
            Config.logger.error(f"Failed to send JSON report to {endpoint}: {e}")
        return None

def send_report_with_response(endpoint, data=None, files=None, params=None, headers=None):
    """Send report and return the response object"""
    url = f"{Config.API_BASE_URL}{endpoint}"
    try:
        response = requests.post(url, data=data, files=files, params=params, headers=headers)
        response.raise_for_status()
        Config.logger.info(f"Successfully sent report to {endpoint}")
        return response
    except requests.RequestException as e:
        Config.logger.error(f"Failed to send report to {endpoint}: {e}")
        return None