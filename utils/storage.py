"""
Data storage and persistence utilities.
"""
import json
import os
from typing import List
from config.settings import REGISTERED_OBJECTS_FILE

class Storage:
    """Handles data persistence."""

    @staticmethod
    def load_registered_objects() -> List[str]:
        if os.path.exists(REGISTERED_OBJECTS_FILE):
            try:
                with open(REGISTERED_OBJECTS_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading registered objects: {e}")
                return []
        return []

    @staticmethod
    def save_registered_objects(objects: List[str]) -> bool:
        try:
            os.makedirs(os.path.dirname(REGISTERED_OBJECTS_FILE), exist_ok=True)
            with open(REGISTERED_OBJECTS_FILE, 'w') as f:
                json.dump(objects, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving registered objects: {e}")
            return False
