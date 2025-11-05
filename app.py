"""
VisionAssist - COMPLETE CLEAN APP.PY
Copy this ENTIRE file and replace your app.py
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

from config.settings import RTSP_FRAME_SKIP, COCO_CLASSES
from core.detection import ObjectDetector
from core.spatial import SpatialAnalyzer
from services.ai_service import AIAssistant
from services.speech_service import SpeechService
from utils.storage import Storage
from ui.sidebar import Sidebar
from ui.voice_assistant import VoiceAssistantUI

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(page_title="VisionAssist Lost-Found AI", page_icon="👁️", layout="wide")

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_session_state():
    if 'registered_objects' not in st.session_state:
        st.session_state.registered_objects = Storage.load_registered_objects()
    if 'rtsp_streaming' not in st.session_state:
        st.session_state.rtsp_streaming = False
    if 'last_detections' not in st.session_state:
        st.session_state.last_detections = []
    if 'last_image' not in st.session_state:
        st.session_state.last_image = None
    if 'stream_final_frame' not in st.session_state:
        st.session_state.stream_final_frame = None
    if 'stream_final_detections' not in st.session_state:
        st.session_state.stream_final_detections = []
    if 'show_stream_summary' not in st.session_state:
        st.session_state.show_stream_summary = False
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'result_announced' not in st.session_state:
        st.session_state.result_announced = False
    if 'target_object_from_voice' not in st.session_state:
        st.session_state.target_object_from_voice = None
    if 'voice_ui_instance' not in st.session_state:
        st.session_state.voice_ui_instance = None
    if 'last_uploaded_file_name' not in st.session_state:
        st.session_state.last_uploaded_file_name = None

initialize_session_state()

# Load services
detector = ObjectDetector()
ai_assistant = AIAssistant()

# Create voice UI instance once
if st.session_state.voice_ui_instance is None:
    st.session_state.voice_ui_instance = VoiceAssistantUI(ai_assistant)

voice_ui = st.session_state.voice_ui_instance

# ============================================================================
# PAGE TITLE
# ============================================================================

st.title("👁️ VisionAssist Lost-Found AI")
st.markdown("**Your intelligent assistant for finding objects with AI**")

# ============================================================================
# SIDEBAR
# ============================================================================

Sidebar.render_audio_control()
Sidebar.render_registered_objects()
target_object = Sidebar.render_object_finder()

# ============================================================================
# INPUT SOURCE (OUTSIDE COLUMNS)
# ============================================================================

st.header("📷 Input Source")
input_source = st.radio(
    "Choose input source:",
    ["Upload Image", "RTSP Stream"],
    key="input_source_radio_main",
    horizontal=False
)

# Clear data when switching sources
if input_source == "Upload Image":
    st.session_state.stream_final_frame = None
    st.session_state.stream_final_detections = []
    st.session_state.show_stream_summary = False

if input_source == "RTSP Stream":
    st.session_state.last_image = None
    st.session_state.last_detections = []

# ============================================================================
# MAIN CONTENT - TWO COLUMNS
# ============================================================================

col1, col2 = st.columns([1, 1])

# ========== LEFT COLUMN ==========
with col1:
    if input_source == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # if st.button("🔍 Detect Objects", type="primary", key="manual_detect"):
            with st.spinner("🔍 Detecting objects..."):
                image_array = np.array(image)
                annotated_image, detections = detector.detect(image_array, target_object)
                st.session_state.last_image = annotated_image
                st.session_state.last_detections = detections
                st.success(f"✅ Detected {len(detections)} objects")
    
    elif input_source == "RTSP Stream":
        st.markdown("**📹 Connect to RTSP Camera**")
        rtsp_url = st.text_input("RTSP URL", placeholder="rtsp://username:password@192.168.1.100:554/stream")
        
        col_btn1, col_btn2 = st.columns(2)
        start_stream = col_btn1.button("▶️ Start Stream", type="primary")
        stop_stream = col_btn2.button("⏹️ Stop Stream")
        
        if stop_stream:
            st.session_state.rtsp_streaming = False
            st.session_state.show_stream_summary = True
        
        if start_stream and rtsp_url:
            st.session_state.show_stream_summary = False
            st.session_state.result_announced = False
            st.session_state.rtsp_streaming = True
            
            with st.spinner("🔗 Connecting to RTSP stream..."):
                cap = cv2.VideoCapture(rtsp_url)
                
                if not cap.isOpened():
                    st.error("❌ Failed to connect to RTSP stream")
                    st.session_state.rtsp_streaming = False
                else:
                    st.success("✅ Connected! Stream is running...")
                    stream_placeholder = st.empty()
                    detection_info = st.empty()
                    frame_count = 0
                    
                    try:
                        while cap.isOpened() and st.session_state.rtsp_streaming:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            if frame_count % RTSP_FRAME_SKIP == 0:
                                annotated_frame, detections = detector.detect(frame, target_object)
                                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                                
                                st.session_state.stream_final_frame = annotated_frame_rgb
                                st.session_state.stream_final_detections = detections
                                
                                stream_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
                                
                                if detections:
                                    detection_text = f"**{len(detections)} objects detected:**\n"
                                    for det in detections:
                                        detection_text += f"• {det['class']} ({det['confidence']:.2f})\n"
                                    detection_info.markdown(detection_text)
                            
                            frame_count += 1
                            if not st.session_state.rtsp_streaming:
                                break
                            time.sleep(0.01)
                    except Exception as e:
                        st.error(f"❌ Stream error: {str(e)}")
                    finally:
                        cap.release()
                        st.session_state.rtsp_streaming = False
                        st.session_state.show_stream_summary = True
        
        if st.session_state.show_stream_summary and st.session_state.stream_final_frame is not None:
            st.markdown("---")
            st.header("📋 Final Results")
            st.image(st.session_state.stream_final_frame, caption="Last Frame", use_container_width=True)
            
            if st.session_state.stream_final_detections:
                st.success(f"✅ {len(st.session_state.stream_final_detections)} object(s)")
                if target_object:
                    found = any(
                        target_object.lower() in d['class'].lower()
                        for d in st.session_state.stream_final_detections
                    )
                    if found:
                        spatial_desc = SpatialAnalyzer.generate_description(
                            target_object,
                            st.session_state.stream_final_detections,
                            st.session_state.stream_final_frame.shape
                        )
                        st.success(f"🎯 {spatial_desc}")
                        if not st.session_state.result_announced:
                            SpeechService.speak(f"Result: {spatial_desc}")
                            st.session_state.result_announced = True

# ========== RIGHT COLUMN ==========
with col2:
    if input_source == "Upload Image":
        if st.session_state.last_image is not None:
            st.header("📊 Results")
            st.image(st.session_state.last_image, caption="Detection Results", use_container_width=True)
            
            if st.session_state.last_detections:
                st.success(f"✅ Found {len(st.session_state.last_detections)} object(s)!")
                for i, detection in enumerate(st.session_state.last_detections):
                    st.write(f"**{i+1}. {detection['class']}** - {detection['confidence']:.2f}")
                
                if target_object:
                    found = any(
                        target_object.lower() in d['class'].lower()
                        for d in st.session_state.last_detections
                    )
                    if found:
                        spatial_desc = SpatialAnalyzer.generate_description(
                            target_object,
                            st.session_state.last_detections,
                            st.session_state.last_image.shape
                        )
                        st.success(f"🎯 {spatial_desc}")
                        SpeechService.speak(spatial_desc)
        else:
            st.header("📊 Results")
            st.info("Upload an image to see results")
    
    # Voice Assistant
    voice_ui.render()

st.markdown("---")
st.markdown("*VisionAssist - Powered by Perplexity AI & YOLO11*")
