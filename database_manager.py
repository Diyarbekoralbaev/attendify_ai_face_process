# database_manager.py

import threading
import numpy as np
import faiss
from pymongo import MongoClient
from datetime import datetime
from config import Config
import os

from funcs import compute_sim


class DatabaseManager:
    def __init__(self):
        self.mongo_client = MongoClient(os.getenv('MONGODB_LOCAL'))
        self.mongo_db = self.mongo_client.empl_time_fastapi
        self.employees_collection = self.mongo_db.employees
        self.clients_collection = self.mongo_db.clients

        # Initialize Faiss indexes with Inner Product for cosine similarity
        self.DIMENSIONS = Config.DIMENSIONS
        self.faiss_index_employee = faiss.IndexIDMap(faiss.IndexFlatIP(self.DIMENSIONS))
        self.faiss_index_client = faiss.IndexIDMap(faiss.IndexFlatIP(self.DIMENSIONS))
        self.lock = threading.Lock()

        # Maintain a mapping from person_id to embedding
        self.employee_embeddings_map = {}
        self.client_embeddings_map = {}

        self.load_faiss_indexes()

    def load_faiss_indexes(self):
        with self.lock:
            Config.logger.info("Loading Faiss indexes for employees and clients.")

            # Reset the indexes
            self.faiss_index_employee = faiss.IndexIDMap(faiss.IndexFlatIP(self.DIMENSIONS))
            self.faiss_index_client = faiss.IndexIDMap(faiss.IndexFlatIP(self.DIMENSIONS))
            self.employee_embeddings_map = {}
            self.client_embeddings_map = {}

            # Load employee embeddings
            employee_embeddings = []
            employee_ids = []
            for emp in self.employees_collection.find({"embedding": {"$exists": True}}):
                embedding = np.array(emp['embedding']).astype('float32')
                if embedding.shape[0] != self.DIMENSIONS:
                    Config.logger.warning(f"Employee ID {emp['person_id']} has invalid embedding shape.")
                    continue
                norm = np.linalg.norm(embedding)
                if norm == 0:
                    Config.logger.warning(f"Employee ID {emp['person_id']} has zero norm embedding.")
                    continue
                embedding = embedding / norm  # Normalize for cosine similarity
                employee_embeddings.append(embedding)
                employee_ids.append(emp['person_id'])
                self.employee_embeddings_map[emp['person_id']] = embedding

            if employee_embeddings:
                employee_embeddings = np.array(employee_embeddings)
                faiss.normalize_L2(employee_embeddings)  # Ensure normalization
                self.faiss_index_employee.add_with_ids(employee_embeddings, np.array(employee_ids))
                Config.logger.info(f"Loaded {len(employee_embeddings)} employee embeddings into Faiss index.")
            else:
                Config.logger.warning("No employee embeddings loaded into Faiss index.")

            # Load client embeddings
            client_embeddings = []
            client_ids = []
            for cli in self.clients_collection.find({"embedding": {"$exists": True}}):
                embedding = np.array(cli['embedding']).astype('float32')
                if embedding.shape[0] != self.DIMENSIONS:
                    Config.logger.warning(f"Client ID {cli['person_id']} has invalid embedding shape.")
                    continue
                norm = np.linalg.norm(embedding)
                if norm == 0:
                    Config.logger.warning(f"Client ID {cli['person_id']} has zero norm embedding.")
                    continue
                embedding = embedding / norm  # Normalize for cosine similarity
                client_embeddings.append(embedding)
                client_ids.append(cli['person_id'])
                self.client_embeddings_map[cli['person_id']] = embedding

            if client_embeddings:
                client_embeddings = np.array(client_embeddings)
                faiss.normalize_L2(client_embeddings)  # Ensure normalization
                self.faiss_index_client.add_with_ids(client_embeddings, np.array(client_ids))
                Config.logger.info(f"Loaded {len(client_embeddings)} client embeddings into Faiss index.")
            else:
                Config.logger.warning("No client embeddings loaded into Faiss index.")

    def add_employee_embedding(self, person_id, embedding):
        with self.lock:
            # norm = np.linalg.norm(embedding)
            # if norm == 0:
            #     Config.logger.error(f"Cannot add employee {person_id} with zero norm embedding.")
            #     return
            # embedding = embedding / norm
            self.employees_collection.update_one(
                {"person_id": person_id},
                {"$set": {
                    "embedding": embedding.tolist(),
                    "updated_at": datetime.now()
                }},
                upsert=True
            )
            self.faiss_index_employee.add_with_ids(
                np.array([embedding]).astype('float32'),
                np.array([person_id], dtype='int64')
            )
            self.employee_embeddings_map[person_id] = embedding
            Config.logger.info(f"Stored/Updated embedding for Employee ID: {person_id}")

    def add_client_embedding(self, person_id, embedding):
        with self.lock:
            norm = np.linalg.norm(embedding)
            if norm == 0:
                Config.logger.error(f"Cannot add client {person_id} with zero norm embedding.")
                return
            embedding = embedding / norm
            self.clients_collection.update_one(
                {"person_id": person_id},
                {"$set": {
                    "embedding": embedding.tolist(),
                    "updated_at": datetime.now()
                }},
                upsert=True
            )
            self.faiss_index_client.add_with_ids(
                np.array([embedding]).astype('float32'),
                np.array([person_id], dtype='int64')
            )
            self.client_embeddings_map[person_id] = embedding
            Config.logger.info(f"Stored/Updated embedding for Client ID: {person_id}")

    def remove_employee_embedding(self, person_id):
        with self.lock:
            self.employees_collection.delete_one({"person_id": person_id})
            self.employee_embeddings_map.pop(person_id, None)
            try:
                self.faiss_index_employee.remove_ids(np.array([person_id], dtype='int64'))
                Config.logger.info(f"Removed embedding for Employee ID: {person_id}")
            except Exception as e:
                Config.logger.error(f"Error removing embedding from Faiss index: {e}")

    def remove_client_embedding(self, person_id):
        with self.lock:
            self.clients_collection.delete_one({"person_id": person_id})
            self.client_embeddings_map.pop(person_id, None)
            try:
                self.faiss_index_client.remove_ids(np.array([person_id], dtype='int64'))
                Config.logger.info(f"Removed embedding for Client ID: {person_id}")
            except Exception as e:
                Config.logger.error(f"Error removing embedding from Faiss index: {e}")

    def remove_deleted_employees(self, fetched_employee_ids):
        with self.lock:
            deleted_employees = self.employees_collection.find({"person_id": {"$nin": fetched_employee_ids}})
            deleted_employee_ids = [emp['person_id'] for emp in deleted_employees]

            if deleted_employee_ids:
                try:
                    self.employees_collection.delete_many({"person_id": {"$in": deleted_employee_ids}})
                    Config.logger.info(f"Removed deleted employees: {deleted_employee_ids}")
                    # Rebuild Faiss index
                    self.load_faiss_indexes()
                except Exception as e:
                    Config.logger.error(f"Error removing deleted employees: {e}")

    def remove_deleted_clients(self, fetched_client_ids):
        with self.lock:
            deleted_clients = self.clients_collection.find({"person_id": {"$nin": fetched_client_ids}})
            deleted_client_ids = [cli['person_id'] for cli in deleted_clients]

            if deleted_client_ids:
                try:
                    self.clients_collection.delete_many({"person_id": {"$in": deleted_client_ids}})
                    Config.logger.info(f"Removed deleted clients: {deleted_client_ids}")
                    # Rebuild Faiss index
                    self.load_faiss_indexes()
                except Exception as e:
                    Config.logger.error(f"Error removing deleted clients: {e}")


    def find_matching_employee(self, embedding):
        with self.lock:
            if not self.employee_embeddings_map:
                return None, 0

            max_similarity = -1
            best_employee = None

            for person_id, stored_embedding in self.employee_embeddings_map.items():
                similarity = compute_sim(embedding, stored_embedding)
                if similarity is not None and similarity > max_similarity:
                    max_similarity = similarity
                    best_employee = self.employees_collection.find_one({"person_id": person_id})

            if max_similarity > Config.EMPLOYEE_SIMILARITY_THRESHOLD:
                return best_employee, max_similarity
            else:
                return None, 0

    def find_matching_client(self, embedding):
        with self.lock:
            if not self.client_embeddings_map:
                return None, 0

            max_similarity = -1
            best_client = None

            for person_id, stored_embedding in self.client_embeddings_map.items():
                similarity = compute_sim(embedding, stored_embedding)
                if similarity is not None and similarity > max_similarity:
                    max_similarity = similarity
                    best_client = self.clients_collection.find_one({"person_id": person_id})

            if max_similarity > Config.CHECK_NEW_CLIENT:
                return best_client, max_similarity
            else:
                return None, 0
