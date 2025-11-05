"""
Sidebar UI components.
"""
import streamlit as st
from typing import Optional
from config.settings import COCO_CLASSES, COMMON_OBJECTS
from services.speech_service import SpeechService
from utils.storage import Storage

class Sidebar:
    """Manages sidebar UI components."""


    @staticmethod
    def render_audio_control():
        """Render audio control section."""
        st.sidebar.markdown("---")
        st.sidebar.header("🔊 Audio Control")
        
        if st.sidebar.button("⏹️ Stop Audio", key="stop_audio_global"):
            try:
                # Try to stop audio safely
                if SpeechService.stop_audio():
                    st.sidebar.success("✅ Audio stopped")
                else:
                    st.sidebar.info("ℹ️ No audio playing")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Could not stop audio: {str(e)}")

    
    @staticmethod
    def render_registered_objects():
        """Render registered objects management section."""
        st.sidebar.markdown("---")
        st.sidebar.header("📋 Registered Objects")
        
        with st.sidebar.expander("➕ Register New Object"):
            selected_class = st.selectbox(
                "Choose object class:",
                [""] + COCO_CLASSES,
                key="coco_select"
            )
            if st.button("Register Object") and selected_class:
                if selected_class not in st.session_state.registered_objects:
                    st.session_state.registered_objects.append(selected_class)
                    Storage.save_registered_objects(st.session_state.registered_objects)
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
                    Storage.save_registered_objects(st.session_state.registered_objects)
                    st.rerun()
        else:
            st.sidebar.info("No objects registered yet")
    
    @staticmethod
    def render_object_finder() -> Optional[str]:
        """
        Render object finder section.
        Only includes: Type, Browse common, or Select from registered.
        Voice input is handled separately in voice assistant.
        
        Returns:
            Selected target object or None
        """
        st.sidebar.markdown("---")
        st.sidebar.header("🎯 Object to Find")
        
        input_method = st.sidebar.radio(
            "How would you like to specify the object?",
            [
                "Type object name",
                "Browse common objects",
                "Select from registered"
            ]
        )
        
        target_object = None
        
        if input_method == "Type object name":
            target_object = st.sidebar.text_input(
                "Enter object name:",
                placeholder="e.g., keys, wallet, phone"
            )
        
        elif input_method == "Browse common objects":
            target_object = st.sidebar.selectbox(
                "Select object:",
                [""] + COMMON_OBJECTS
            )
        
        elif input_method == "Select from registered":
            if st.session_state.registered_objects:
                target_object = st.sidebar.selectbox(
                    "Select registered object:",
                    [""] + st.session_state.registered_objects
                )
            else:
                st.sidebar.warning("No registered objects. Register first.")
        
        return target_object
