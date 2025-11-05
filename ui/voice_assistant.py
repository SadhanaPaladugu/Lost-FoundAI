import streamlit as st
from services.ai_service import AIAssistant
from services.speech_service import SpeechService
from core.spatial import SpatialAnalyzer
from config.settings import COCO_CLASSES
from utils.helpers import extract_object_from_query

class VoiceAssistantUI:
    """Voice assistant interface."""
    
    def __init__(self, ai_assistant: AIAssistant):
        self.ai_assistant = ai_assistant

    def render(self):
        """Render voice assistant interface."""
        st.header("🔊 AI Voice Assistant")
        
        if not self.ai_assistant.is_available():
            st.warning("⚠️ AI assistant not configured")
            st.info("Set PERPLEXITY_API_KEY environment variable to enable")
            return
        
        st.markdown("**Ask me anything!**")
        st.caption("Try: 'Where is my phone?', 'What can you do?'")
        
        col_q1, col_q2 = st.columns([3, 1])
        question = col_q1.text_input(
            "Question:",
            placeholder="Ask me anything...",
            label_visibility="collapsed",
            key="voice_question_input"
        )
        
        voice_button_clicked = False
        if SpeechService.is_stt_available():
            if col_q2.button("🎤 Voice", key="voice_button"):
                voice_button_clicked = True
        else:
            col_q2.info("Text only")
        
        # Handle voice input
        if voice_button_clicked:
            try:
                with st.spinner("🎤 Listening... Speak now!"):
                    voice_query = SpeechService.listen()
                    if voice_query:
                        st.success(f"✅ Heard: {voice_query}")
                        question = voice_query
                    else:
                        st.warning("❌ No speech detected. Please try again.")
                        return
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                return
        
        # Process the question if not empty
        if question and question.strip():
            self._process_query(question)
        
        # Show conversation history
        self._render_history()

    def _process_query(self, question: str):
        """Process user query."""
        st.session_state.conversation_history.append({
            "role": "user",
            "content": question
        })
        
        # Extract object from question
        object_of_interest = extract_object_from_query(question, COCO_CLASSES)
        
        if object_of_interest:
            # Object finding query
            response = self._handle_voice_object_query(object_of_interest)
        else:
            # General AI query
            current_detections = (
                st.session_state.stream_final_detections
                if st.session_state.stream_final_detections
                else st.session_state.last_detections
            )
            try:
                response = self.ai_assistant.query(question, current_detections)
            except Exception as e:
                response = f"Error: {str(e)}"
        
        st.info(f"💬 {response}")
        try:
            SpeechService.speak(response)
        except Exception as e:
            st.warning(f"⚠️ Could not speak: {str(e)}")
        
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": response
        })

    def _handle_voice_object_query(self, object_name: str) -> str:
        """Handle object finding query from voice."""
        # Check if we have image with detections
        if (st.session_state.last_image is not None and 
            st.session_state.last_detections):
            image_to_use = st.session_state.last_image
            detections_to_use = st.session_state.last_detections
        # Check if we have stream with detections
        elif (st.session_state.stream_final_frame is not None and 
              st.session_state.stream_final_detections):
            image_to_use = st.session_state.stream_final_frame
            detections_to_use = st.session_state.stream_final_detections
        # No data available
        else:
            return f"To find the {object_name}, please upload an image or provide an RTSP stream URL."
        
        # Search for object
        found = any(
            object_name.lower() in det['class'].lower()
            for det in detections_to_use
        )
        
        if found:
            return SpatialAnalyzer.generate_description(
                object_name,
                detections_to_use,
                image_to_use.shape
            )
        else:
            return f"I analyzed the image but couldn't find the {object_name}."

    def _render_history(self):
        """Render conversation history."""
        if st.session_state.conversation_history:
            st.markdown("---")
            st.subheader("💬 Recent Chat")
            for msg in st.session_state.conversation_history[-3:]:
                if msg['role'] == 'user':
                    st.markdown(f"**You:** {msg['content']}")
                else:
                    st.markdown(f"**VisionAssist:** {msg['content']}")
