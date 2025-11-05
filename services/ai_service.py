"""
AI assistant service using Perplexity API.
"""
from typing import List, Dict, Optional
from openai import OpenAI
from config.settings import (
    PERPLEXITY_API_KEY,
    PERPLEXITY_BASE_URL,
    PERPLEXITY_MODEL,
    MAX_TOKENS,
    TEMPERATURE
)

class AIAssistant:
    """Handles conversational AI interactions."""

    def __init__(self):
        self.client = self._initialize_client()

    def _initialize_client(self) -> Optional[OpenAI]:
        if not PERPLEXITY_API_KEY:
            return None
        try:
            return OpenAI(
                api_key=PERPLEXITY_API_KEY,
                base_url=PERPLEXITY_BASE_URL
            )
        except Exception as e:
            print(f"Error initializing AI client: {e}")
            return None

    def is_available(self) -> bool:
        return self.client is not None

    def query(
        self,
        user_query: str,
        detected_objects: Optional[List[Dict]] = None
    ) -> str:
        """Process user query with AI assistant."""
        if not self.client:
            return "AI assistant not configured. Please set PERPLEXITY_API_KEY environment variable."
        system_message = self._build_system_message(detected_objects)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_query}
        ]
        try:
            response = self.client.chat.completions.create(
                model=PERPLEXITY_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error communicating with AI: {str(e)}"

    def _build_system_message(self, detected_objects: Optional[List[Dict]]) -> str:
        base_message = """You are VisionAssist, an AI-powered vision assistant for visually impaired users.
Your capabilities:
- You detect 80 object types using YOLO11 including: remote, phone, keys, glasses, cup, bottle, chair, couch, bed, and many more.
- You analyze images and camera feeds to find objects
- You give spatial descriptions like "the remote is near the couch"
- You support RTSP camera feeds
Guidelines:
- Be friendly, concise, supportive
- Keep responses under 3 sentences when possible
"""
        if detected_objects:
            objects_list = ", ".join([obj['class'] for obj in detected_objects])
            base_message += f"\n\nCurrently detected objects in the image: {objects_list}"
        return base_message
