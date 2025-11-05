"""
General helper utilities.
"""
from typing import List
import warnings
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

def extract_object_from_query(query: str, coco_classes: List[str]) -> str:
    tokens = query.lower().replace('?', '').replace(',', '').replace('.', '').split()
    for token in tokens:
        if token in [c.lower() for c in coco_classes]:
            return token
    return ""

def format_confidence(confidence: float) -> str:
    return f"{confidence:.2%}"

def truncate_text(text: str, max_length: int = 100) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
