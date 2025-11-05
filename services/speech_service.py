"""
Speech services for text-to-speech and speech-to-text.
"""
import os
import tempfile
import time
from contextlib import redirect_stderr
from typing import Optional
from config.settings import (
    SPEECH_TIMEOUT,
    PHRASE_TIME_LIMIT,
    AMBIENT_NOISE_DURATION
)

SPEECH_AVAILABLE = False
MICROPHONE_AVAILABLE = False

try:
    from gtts import gTTS
    import pygame
    pygame.mixer.init()
    SPEECH_AVAILABLE = True
except ImportError:
    pass

try:
    import speech_recognition as sr
    try:
        import pyaudio
        MICROPHONE_AVAILABLE = True
    except (ImportError, OSError):
        MICROPHONE_AVAILABLE = False
except ImportError:
    pass

class SpeechService:
    """Handles text-to-speech and speech-to-text."""
    @staticmethod
    def is_tts_available() -> bool:
        return SPEECH_AVAILABLE

    @staticmethod
    def is_stt_available() -> bool:
        return MICROPHONE_AVAILABLE

    @staticmethod
    def speak(text: str) -> None:
        if not SPEECH_AVAILABLE:
            print(f"Speech output: {text}")
            return
        try:
            tts = gTTS(text=text, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                pygame.mixer.music.load(tmp_file.name)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                pygame.mixer.music.unload()
                os.unlink(tmp_file.name)
        except Exception as e:
            print(f"Text-to-speech error: {e}")

    @staticmethod
    def stop_audio() -> bool:
        """
        Stop currently playing audio safely.
        
        Returns:
            True if audio was stopped, False otherwise
        """
        if not SPEECH_AVAILABLE:
            return False
        
        try:
            import pygame
            # Check if mixer is initialized
            if pygame.mixer.get_init():
                # Stop music safely
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                    return True
            return False
        except Exception as e:
            print(f"Error stopping audio: {e}")
            return False


    @staticmethod
    def listen() -> Optional[str]:
        if not MICROPHONE_AVAILABLE:
            return None
        try:
            r = sr.Recognizer()
            with redirect_stderr(open(os.devnull, 'w')):
                with sr.Microphone() as source:
                    r.adjust_for_ambient_noise(source, duration=AMBIENT_NOISE_DURATION)
                    audio = r.listen(source, timeout=SPEECH_TIMEOUT, phrase_time_limit=PHRASE_TIME_LIMIT)
            text = r.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except Exception as e:
            print(f"Speech recognition error: {e}")
            return None
