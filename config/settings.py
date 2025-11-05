"""
Configuration settings for VisionAssist application.
"""
import os
from typing import List

# Model Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "yolo11n.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))

# API Configuration
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
PERPLEXITY_MODEL = "sonar"
MAX_TOKENS = 150
TEMPERATURE = 0.7

REGISTERED_OBJECTS_FILE = "data/registered_objects.json"
LOGS_DIR = "logs"

SPEECH_TIMEOUT = 5
PHRASE_TIME_LIMIT = 10
AMBIENT_NOISE_DURATION = 0.5

RTSP_FRAME_SKIP = 5  # Process every 5th frame
SPATIAL_DISTANCE_THRESHOLD = 200
SPATIAL_POSITION_THRESHOLD = 50

COCO_CLASSES: List[str] = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
COMMON_OBJECTS: List[str] = [
    "keys", "wallet", "phone", "glasses", "watch", "remote", 
    "book", "cup", "bottle", "bag"
]

# Make sure required directories exist
os.makedirs(os.path.dirname(REGISTERED_OBJECTS_FILE), exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
