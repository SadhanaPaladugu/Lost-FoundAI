
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import io
import time
import json

# For speech functionality - make PyAudio optional
SPEECH_AVAILABLE = False
MICROPHONE_AVAILABLE = False

try:
    from gtts import gTTS
    import pygame
    SPEECH_AVAILABLE = True  # Text-to-speech works
except ImportError:
    pass

try:
    import speech_recognition as sr
    MICROPHONE_AVAILABLE = True  # Microphone input works
except ImportError:
    pass

if not SPEECH_AVAILABLE:
    st.warning("⚠️ Text-to-speech not available. Install: pip install gTTS pygame")
if not MICROPHONE_AVAILABLE:
    st.info("ℹ️ Microphone input not available. You can still use text input.")

MODEL_PATH = "yolo11n.pt"
REGISTERED_OBJECTS_FILE = "registered_objects.json"

# COCO 80 classes
COCO_CLASSES = [
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

def load_registered_objects():
    if os.path.exists(REGISTERED_OBJECTS_FILE):
        with open(REGISTERED_OBJECTS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_registered_objects(objects):
    with open(REGISTERED_OBJECTS_FILE, 'w') as f:
        json.dump(objects, f)

@st.cache_resource
def load_model():
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO11n model: {e}")
        st.info("Make sure yolo11n.pt is in the same directory as app.py")
        return None

def speak_text(text):
    """Convert text to speech - works even without PyAudio"""
    if not SPEECH_AVAILABLE:
        st.info(f"🔊 Speech output: {text}")
        return

    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts.save(tmp_file.name)
            pygame.mixer.init()
            pygame.mixer.music.load(tmp_file.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            os.unlink(tmp_file.name)
    except Exception as e:
        st.warning(f"Text-to-speech error: {e}")
        st.info(f"🔊 Would say: {text}")

def listen_for_speech():
    """Listen for speech input - requires PyAudio"""
    if not MICROPHONE_AVAILABLE:
        st.error("❌ Microphone input requires PyAudio")
        st.info("💡 Use text input instead, or try these fixes:")
        st.code("""# On macOS:
brew install portaudio
pip3 install pyaudio

# Or alternative:
pip3 install --upgrade pip setuptools wheel
pip3 install pyaudio

# Test with:
python3 -c "import pyaudio; print('PyAudio OK!')"
""")
        return None

    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now!")
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source, timeout=5)
        text = r.recognize_google(audio)
        return text
    except sr.WaitTimeoutError:
        st.warning("No speech detected within timeout period")
        return None
    except sr.UnknownValueError:
        st.warning("Could not understand audio")
        return None
    except Exception as e:
        st.error(f"Speech recognition error: {e}")
        return None

def get_spatial_relationship(bbox1, bbox2, name1, name2):
    x1_center = (bbox1[0] + bbox1[2]) / 2
    y1_center = (bbox1[1] + bbox1[3]) / 2
    x2_center = (bbox2[0] + bbox2[2]) / 2
    y2_center = (bbox2[1] + bbox2[3]) / 2
    dx = x2_center - x1_center
    dy = y2_center - y1_center
    relationships = []
    if abs(dy) > 50:
        if dy > 0:
            relationships.append("above")
        else:
            relationships.append("below")
    if abs(dx) > 50:
        if dx > 0:
            relationships.append("to the left of")
        else:
            relationships.append("to the right of")
    distance = np.sqrt(dx**2 + dy**2)
    if distance < 200:
        return f"near the {name2}"
    elif relationships:
        return " and ".join(relationships) + f" the {name2}"
    else:
        return f"near the {name2}"

def generate_spatial_description(target_object, all_detections, image_shape):
    if not all_detections:
        return "No objects detected to describe location."
    target_det = None
    other_objects = []
    for det in all_detections:
        if target_object and target_object.lower() in det['class'].lower():
            target_det = det
        else:
            other_objects.append(det)
    if not target_det:
        return f"Could not find {target_object} in the image."
    if not other_objects:
        bbox = target_det['bbox']
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        img_height, img_width = image_shape[:2]
        h_pos = "center"
        if x_center < img_width * 0.33:
            h_pos = "left side"
        elif x_center > img_width * 0.67:
            h_pos = "right side"
        v_pos = "middle"
        if y_center < img_height * 0.33:
            v_pos = "top"
        elif y_center > img_height * 0.67:
            v_pos = "bottom"
        return f"The {target_det['class']} is in the {v_pos} {h_pos} of the image."
    target_bbox = target_det['bbox']
    target_center = np.array([(target_bbox[0] + target_bbox[2]) / 2, 
                              (target_bbox[1] + target_bbox[3]) / 2])
    closest_obj = None
    min_distance = float('inf')
    for obj in other_objects:
        obj_bbox = obj['bbox']
        obj_center = np.array([(obj_bbox[0] + obj_bbox[2]) / 2, 
                               (obj_bbox[1] + obj_bbox[3]) / 2])
        distance = np.linalg.norm(target_center - obj_center)
        if distance < min_distance:
            min_distance = distance
            closest_obj = obj
    if closest_obj:
        relationship = get_spatial_relationship(
            target_det['bbox'], 
            closest_obj['bbox'],
            target_det['class'],
            closest_obj['class']
        )
        return f"The {target_det['class']} is {relationship}."
    return f"Found the {target_det['class']}."

def detect_objects(model, image, target_object=None):
    if model is None:
        return image, []
    results = model(image)
    detections = []
    annotated_image = image.copy()
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': (int(x1), int(y1), int(x2), int(y2))
                })
                if target_object and target_object.lower() in class_name.lower():
                    color = (0, 255, 0)
                    thickness = 3
                else:
                    color = (255, 165, 0)
                    thickness = 2
                cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(annotated_image, label, (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated_image, detections

def main():
    st.set_page_config(page_title="VisionAssist Lost-Found AI", page_icon="👁️", layout="wide")

    # Initialize session state
    if 'registered_objects' not in st.session_state:
        st.session_state.registered_objects = load_registered_objects()
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

    st.title("👁️ Vision/Voice Assist Lost-Found AI")
    st.markdown("**Helping users locate objects using AI-powered vision**")

    with st.expander("♿ Accessibility Information"):
        status_text = ""
        if SPEECH_AVAILABLE:
            status_text += "✅ Text-to-speech: Available\n"
        else:
            status_text += "❌ Text-to-speech: Not available\n"

        if MICROPHONE_AVAILABLE:
            status_text += "✅ Microphone input: Available\n"
        else:
            status_text += "⚠️ Microphone input: Not available (use text input)\n"

        st.markdown(status_text)
        st.markdown("""
        - **Keyboard Navigation**: Tab to navigate, Enter to activate
        - **Screen Reader Support**: All elements labeled
        - **Voice Input**: Speak queries (requires PyAudio)
        - **Text Input**: Alternative to voice (always works)
        - **Audio Output**: Results announced via speech
        - **RTSP Support**: Connect IP camera for live detection
        """)

    model = load_model()
    if model is None:
        st.stop()

    # Sidebar - Registered Objects
    st.sidebar.header("📋 Registered Objects")
    with st.sidebar.expander("➕ Register New Object"):
        selected_class = st.selectbox("Choose object class:", [""] + COCO_CLASSES, key="coco_select")
        if st.button("Register Object") and selected_class:
            if selected_class not in st.session_state.registered_objects:
                st.session_state.registered_objects.append(selected_class)
                save_registered_objects(st.session_state.registered_objects)
                st.success(f"✅ Registered: {selected_class}")
            else:
                st.warning(f"{selected_class} already registered")

    if st.session_state.registered_objects:
        st.sidebar.markdown("**Your registered objects:**")
        for obj in st.session_state.registered_objects:
            col1, col2 = st.sidebar.columns([3, 1])
            col1.write(f"• {obj}")
            if col2.button("❌", key=f"remove_{obj}"):
                st.session_state.registered_objects.remove(obj)
                save_registered_objects(st.session_state.registered_objects)
                st.rerun()
    else:
        st.sidebar.info("No objects registered yet")

    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Object to Find")

    input_method = st.sidebar.radio(
        "How would you like to specify the object?",
        ["Type object name", "Speak object name", "Browse common objects", "Select from registered"]
    )

    target_object = None

    if input_method == "Type object name":
        target_object = st.sidebar.text_input("Enter object name:", placeholder="e.g., keys, wallet, phone")
    elif input_method == "Speak object name":
        if MICROPHONE_AVAILABLE:
            if st.sidebar.button("🎤 Listen for Object Name"):
                with st.spinner("Listening..."):
                    spoken_text = listen_for_speech()
                    if spoken_text:
                        target_object = spoken_text
                        st.sidebar.success(f"Heard: {spoken_text}")
                        speak_text(f"Looking for {spoken_text}")
        else:
            st.sidebar.error("❌ Microphone not available")
            st.sidebar.info("💡 Use 'Type object name' instead")
    elif input_method == "Browse common objects":
        common_objects = ["keys", "wallet", "phone", "glasses", "watch", "remote", "book", "cup", "bottle", "bag"]
        target_object = st.sidebar.selectbox("Select object:", [""] + common_objects)
    elif input_method == "Select from registered":
        if st.session_state.registered_objects:
            target_object = st.sidebar.selectbox("Select registered object:", [""] + st.session_state.registered_objects)
        else:
            st.sidebar.warning("No registered objects. Register first.")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📷 Input Source")
        input_source = st.radio("Choose input source:", ["Upload Image", "RTSP Stream", "Camera (Live)"])

        if input_source == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                st.image(image, caption="Uploaded Image", use_column_width=True)

                if st.button("🔍 Detect Objects", type="primary"):
                    with st.spinner("Analyzing image..."):
                        annotated_image, detections = detect_objects(model, image_array, target_object)
                        st.session_state.last_image = annotated_image
                        st.session_state.last_detections = detections

                        with col2:
                            st.header("📊 Detection Results")
                            st.image(annotated_image, caption="Detection Results", use_column_width=True)

                            if detections:
                                st.success(f"Found {len(detections)} object(s)!")
                                for i, detection in enumerate(detections):
                                    st.write(f"**{i+1}. {detection['class']}** - Confidence: {detection['confidence']:.2f}")

                                if target_object:
                                    if target_object.lower() not in [c.lower() for c in COCO_CLASSES]:
                                        feedback = f"Sorry, I am not trained to find '{target_object}'."
                                        st.error(feedback)
                                        speak_text(feedback)
                                    else:
                                        found_target = any(target_object.lower() in det['class'].lower() for det in detections)
                                        if found_target:
                                            spatial_desc = generate_spatial_description(target_object, detections, image_array.shape)
                                            st.success(f"✅ {spatial_desc}")
                                            speak_text(spatial_desc)
                                        else:
                                            feedback = f"I couldn't find your {target_object}."
                                            st.warning(feedback)
                                            speak_text(feedback)
                            else:
                                feedback = "No objects detected. Try adjusting the image."
                                st.warning(feedback)
                                speak_text(feedback)

        elif input_source == "RTSP Stream":
            st.markdown("**Connect to IP Camera via RTSP**")
            rtsp_url = st.text_input("RTSP URL", placeholder="rtsp://username:password@ip:port/stream_path")

            col_btn1, col_btn2 = st.columns(2)
            start_stream = col_btn1.button("▶️ Start Stream", type="primary")
            stop_stream = col_btn2.button("⏹️ Stop Stream")

            if stop_stream:
                st.session_state.rtsp_streaming = False
                st.session_state.show_stream_summary = True
                st.info("⏹️ Stream stopped by user")

            if start_stream and rtsp_url:
                st.session_state.rtsp_streaming = True
                st.session_state.show_stream_summary = False

                with st.spinner("Connecting to RTSP stream..."):
                    cap = cv2.VideoCapture(rtsp_url)

                    if not cap.isOpened():
                        st.error("❌ Failed to open RTSP stream. Please check:")
                        st.error("• RTSP URL format")
                        st.error("• Camera credentials")
                        st.error("• Network connectivity")
                        st.session_state.rtsp_streaming = False
                    else:
                        st.success("✅ Connected to RTSP stream!")

                        with col2:
                            st.header("📊 Live Detection Results")
                            stream_placeholder = st.empty()
                            detection_info = st.empty()

                        frame_count = 0
                        last_spoken_time = 0

                        while cap.isOpened() and st.session_state.rtsp_streaming:
                            ret, frame = cap.read()

                            if not ret:
                                st.warning("⚠️ Failed to read frame")
                                break

                            if frame_count % 5 == 0:
                                annotated_frame, detections = detect_objects(model, frame, target_object)
                                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                                st.session_state.stream_final_frame = annotated_frame_rgb
                                st.session_state.stream_final_detections = detections

                                stream_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)

                                if detections:
                                    detection_text = f"**Detected {len(detections)} objects:**\n"
                                    for det in detections:
                                        detection_text += f"• {det['class']} ({det['confidence']:.2f})\n"
                                    detection_info.markdown(detection_text)

                                    if target_object:
                                        if target_object.lower() not in [c.lower() for c in COCO_CLASSES]:
                                            if time.time() - last_spoken_time > 10:
                                                feedback = f"Sorry, I am not trained to find '{target_object}'."
                                                detection_info.error(feedback)
                                                speak_text(feedback)
                                                last_spoken_time = time.time()
                                        else:
                                            found_target = any(target_object.lower() in det['class'].lower() for det in detections)
                                            if found_target:
                                                spatial_desc = generate_spatial_description(target_object, detections, frame.shape)
                                                detection_info.success(f"✅ {spatial_desc}")
                                                if time.time() - last_spoken_time > 5:
                                                    speak_text(f"Found! {spatial_desc}")
                                                    last_spoken_time = time.time()
                                else:
                                    detection_info.info("No objects detected")

                            frame_count += 1

                            if not st.session_state.rtsp_streaming:
                                break

                            time.sleep(0.01)

                        cap.release()
                        st.session_state.show_stream_summary = True
                        st.info("✅ Stream ended. See final results below.")

            if st.session_state.show_stream_summary and st.session_state.stream_final_frame is not None:
                with col2:
                    st.markdown("---")
                    st.header("📋 Final Stream Summary")
                    st.image(st.session_state.stream_final_frame, caption="Last Frame", use_column_width=True)

                    if st.session_state.stream_final_detections:
                        st.success(f"✅ Found {len(st.session_state.stream_final_detections)} object(s)")

                        for i, det in enumerate(st.session_state.stream_final_detections):
                            st.write(f"**{i+1}. {det['class']}** - {det['confidence']:.2f}")

                        if target_object:
                            found_target = any(target_object.lower() in det['class'].lower() 
                                             for det in st.session_state.stream_final_detections)
                            if found_target:
                                frame_shape = st.session_state.stream_final_frame.shape
                                spatial_desc = generate_spatial_description(
                                    target_object, 
                                    st.session_state.stream_final_detections, 
                                    frame_shape
                                )
                                st.success(f"🎯 {spatial_desc}")
                                speak_text(f"Stream complete. {spatial_desc}")
                            else:
                                feedback = f"Stream complete. Could not find {target_object}."
                                st.warning(feedback)
                                speak_text(feedback)
                    else:
                        feedback = "Stream complete. No objects detected."
                        st.warning(feedback)
                        speak_text(feedback)

        elif input_source == "Camera (Live)":
            st.info("💡 Local webcam requires additional setup.")

    with col2:
        if input_source == "Upload Image" and st.session_state.last_image is None:
            st.header("📊 Detection Results")
            st.info("Upload an image and click 'Detect Objects'.")

        if input_source != "RTSP Stream" or not st.session_state.rtsp_streaming:
            st.markdown("---")
            st.header("🔊 Voice Assistant")
            st.markdown("**Ask VisionAssist a question:**")

            col_q1, col_q2 = st.columns([3, 1])
            question = col_q1.text_input("Question:", placeholder="e.g., 'where is the bowl?'", label_visibility="collapsed")

            if MICROPHONE_AVAILABLE:
                voice_query_btn = col_q2.button("🎤 Voice")
                if voice_query_btn:
                    with st.spinner("Listening..."):
                        voice_query = listen_for_speech()
                        if voice_query:
                            question = voice_query
                            st.success(f"Heard: {voice_query}")
            else:
                col_q2.info("Text only")

            if question:
                tokens = question.lower().replace('?', '').replace(',', '').replace('.', '').split()
                object_of_interest = None

                for token in tokens:
                    if token in [c.lower() for c in COCO_CLASSES]:
                        object_of_interest = token
                        break

                if not object_of_interest:
                    feedback = "Sorry, I am not trained to find that object."
                    st.error(feedback)
                    speak_text(feedback)
                else:
                    image_to_analyze = None
                    detections_to_use = None

                    if st.session_state.stream_final_frame is not None:
                        image_to_analyze = st.session_state.stream_final_frame
                        detections_to_use = st.session_state.stream_final_detections
                    elif st.session_state.last_image is not None:
                        image_to_analyze = st.session_state.last_image
                        detections_to_use = st.session_state.last_detections

                    if image_to_analyze is not None and detections_to_use:
                        found_target = any(object_of_interest in det['class'].lower() for det in detections_to_use)
                        if found_target:
                            spatial_desc = generate_spatial_description(object_of_interest, detections_to_use, image_to_analyze.shape)
                            st.success(f"✅ {spatial_desc}")
                            speak_text(spatial_desc)
                        else:
                            feedback = f"I couldn't find the {object_of_interest}."
                            st.warning(feedback)
                            speak_text(feedback)
                    else:
                        feedback = "Please upload an image or complete a stream first."
                        st.info(feedback)
                        speak_text(feedback)

    st.markdown("---")
    with st.expander("❓ Help & Troubleshooting"):
        st.markdown("""
        ### PyAudio Installation Issues:

        **macOS:**
        ```bash
        # Install portaudio first
        brew install portaudio

        # Then install PyAudio
        pip3 install pyaudio

        # Or try:
        pip3 install --global-option='build_ext' --global-option='-I/opt/homebrew/include' --global-option='-L/opt/homebrew/lib' pyaudio
        ```

        **Test installation:**
        ```bash
        python3 -c "import pyaudio; print('PyAudio works!')"
        ```

        ### Alternative: Use Text Input
        - Voice input is optional
        - All features work with text input
        - Text-to-speech still works (gTTS + pygame)

        ### Feature Status:
        - ✅ Object detection (always works)
        - ✅ Text input (always works)
        - ✅ Text-to-speech (needs gTTS + pygame)
        - ⚠️ Voice input (needs PyAudio + SpeechRecognition)
        """)

    st.markdown("---")
    st.markdown("*VisionAssist Lost-Found AI - Empowering independence through AI vision*")

if __name__ == "__main__":
    main()
