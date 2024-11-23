# websocket_listener.py

import asyncio
import websockets
import json
from config import Config
from funcs import get_embedding_from_url

async def websocket_listener(db_manager, face_processor):
    uri = f"{Config.API_BASE_URL.replace('http', 'ws')}/ws/"

    async with websockets.connect(uri) as websocket:
        Config.logger.info("Connected to WebSocket server.")
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                Config.logger.info(f"Received data via WebSocket: {data}")

                # Handle the data (e.g., 'employee_update' or 'client_update')
                if data['event'] == 'employee_update' or data['event'] == 'employee_create':
                    await handle_employee_update(data['data'], db_manager, face_processor)
                elif data['event'] == 'employee_delete':
                    await handle_employee_removed(data['data']['id'], db_manager)
                elif data['event'] == 'client_delete':
                    await handle_client_removed(data['data']['id'], db_manager)
                else:
                    Config.logger.warning(f"Unknown data type received: {data['event']}")

            except websockets.ConnectionClosed:
                Config.logger.error("WebSocket connection closed. Reconnecting...")
                await asyncio.sleep(5)  # Wait before reconnecting
                return await websocket_listener(db_manager, face_processor)
            except Exception as e:
                Config.logger.error(f"Error in WebSocket listener: {e}")
                await asyncio.sleep(1)

async def handle_employee_update(employee_data, db_manager, face_processor):
    person_id = employee_data['id']
    image_url = f"{Config.API_BASE_URL}/{employee_data['image']}"
    embedding = get_embedding_from_url(image_url, face_processor)
    if embedding is not None:
        db_manager.add_employee_embedding(person_id, embedding)
        Config.logger.info(f"Updated embedding for Employee ID: {person_id}")
    else:
        Config.logger.error(f"Failed to get embedding for Employee ID: {person_id}")

async def handle_client_update(client_data, db_manager, face_processor):
    person_id = client_data['id']
    image_url = f"{Config.API_BASE_URL}/{client_data['image']}"
    embedding = get_embedding_from_url(image_url, face_processor)
    if embedding is not None:
        db_manager.add_client_embedding(person_id, embedding)
        Config.logger.info(f"Updated embedding for Client ID: {person_id}")
    else:
        Config.logger.error(f"Failed to get embedding for Client ID: {person_id}")

async def handle_employee_removed(employee_id, db_manager):
    person_id = employee_id
    db_manager.remove_employee_embedding(person_id)
    Config.logger.info(f"Removed embedding for Employee ID: {person_id}")


async def handle_client_removed(client_id, db_manager):
    person_id = client_id
    db_manager.remove_client_embedding(person_id)
    Config.logger.info(f"Removed embedding for Client ID: {person_id}")