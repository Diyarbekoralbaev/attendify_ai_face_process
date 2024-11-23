# database_manager.py

import threading
import numpy as np
import faiss
import os
from config import Config

class DatabaseManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.index_path_employee = 'faiss_indexes/faiss_employee.index'
        self.index_path_client = 'faiss_indexes/faiss_client.index'

        # Initialize or load Faiss indexes
        self.faiss_index_employee = self.initialize_index(self.index_path_employee)
        self.faiss_index_client = self.initialize_index(self.index_path_client)

    def initialize_index(self, index_path):
        # Check if the index file exists
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
            Config.logger.info(f"Loaded Faiss index from {index_path}")
        else:
            index = None
            Config.logger.info(f"No existing Faiss index at {index_path}, will create a new one upon first embedding addition.")
        return index

    def save_index(self, index, index_path):
        faiss.write_index(index, index_path)
        Config.logger.info(f"Saved Faiss index to {index_path}")

    def add_embedding(self, person_id, embedding, index_type):
        with self.lock:
            embedding = embedding.astype('float32')
            # Determine dimension from embedding
            d = embedding.shape[0]

            # Initialize the index if it doesn't exist
            index = getattr(self, f'faiss_index_{index_type}')
            if index is None:
                # Wrap the base index with IndexIDMap to support add_with_ids
                base_index = faiss.IndexFlatIP(d)  # Using Inner Product for cosine similarity
                index = faiss.IndexIDMap(base_index)
                setattr(self, f'faiss_index_{index_type}', index)
                Config.logger.info(f"Created new Faiss index for {index_type} with dimension {d}")

            # Normalize the embedding
            faiss.normalize_L2(embedding.reshape(1, -1))

            # Add the embedding with the custom ID
            index.add_with_ids(np.array([embedding]), np.array([person_id], dtype='int64'))
            self.save_index(index, getattr(self, f'index_path_{index_type}'))
            Config.logger.info(f"Added embedding for {index_type.capitalize()} ID: {person_id}")

    def remove_embedding(self, person_id, index_type):
        with self.lock:
            index = getattr(self, f'faiss_index_{index_type}')
            if index is not None:
                try:
                    index.remove_ids(np.array([person_id], dtype='int64'))
                    self.save_index(index, getattr(self, f'index_path_{index_type}'))
                    Config.logger.info(f"Removed embedding for {index_type.capitalize()} ID: {person_id}")
                except Exception as e:
                    Config.logger.error(f"Error removing embedding from Faiss index: {e}")
            else:
                Config.logger.warning(f"No Faiss index found for {index_type} when attempting to remove embedding.")

    def find_matching(self, embedding, index_type, threshold):
        with self.lock:
            index = getattr(self, f'faiss_index_{index_type}')
            if index is None or index.ntotal == 0:
                return None, 0
            embedding = embedding.astype('float32').reshape(1, -1)
            # Normalize the embedding
            faiss.normalize_L2(embedding)
            # Search the index
            distances, indices = index.search(embedding, k=1)
            best_id = indices[0][0]
            max_similarity = distances[0][0]
            if max_similarity > threshold:
                return {'person_id': best_id}, max_similarity
            else:
                return None, 0

    def reset_embeddings(self, index_type):
        with self.lock:
            index = getattr(self, f'faiss_index_{index_type}')
            if index is not None:
                index.reset()
                setattr(self, f'faiss_index_{index_type}', None)
                index_path = getattr(self, f'index_path_{index_type}')
                if os.path.exists(index_path):
                    os.remove(index_path)
                Config.logger.info(f"Reset {index_type} embeddings in Faiss index.")

    # Wrapper methods for employees
    def add_employee_embedding(self, person_id, embedding):
        self.add_embedding(person_id, embedding, 'employee')

    def remove_employee_embedding(self, person_id):
        self.remove_embedding(person_id, 'employee')

    def find_matching_employee(self, embedding):
        return self.find_matching(embedding, 'employee', Config.EMPLOYEE_SIMILARITY_THRESHOLD)

    def reset_employee_embeddings(self):
        self.reset_embeddings('employee')

    # Wrapper methods for clients
    def add_client_embedding(self, person_id, embedding):
        self.add_embedding(person_id, embedding, 'client')

    def remove_client_embedding(self, person_id):
        self.remove_embedding(person_id, 'client')

    def find_matching_client(self, embedding):
        return self.find_matching(embedding, 'client', Config.CHECK_NEW_CLIENT)

    def reset_client_embeddings(self):
        self.reset_embeddings('client')
