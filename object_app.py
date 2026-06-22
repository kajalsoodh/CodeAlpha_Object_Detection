import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

st.set_page_config(page_title="AI Object Detector", page_icon="🔍", layout="centered")

# --- PREMIUM UI CSS ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    .header-container {
        text-align: center;
        padding: 2rem 0rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .main-title {
        font-family: 'Poppins', sans-serif;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class="header-container">
        <h1 class="main-title">🔍 REAL-TIME OBJECT DETECTOR</h1>
        <p style="color: #94a3b8; font-size: 1.1rem;">Computer Vision application using YOLOv8 and OpenCV.</p>
    </div>
    """, unsafe_allow_html=True)

# Cache the model so it doesn't reload on every button press
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # Nano version: lightweight and fast

try:
    model = load_model()
except Exception as e:
    st.error(f"Model load karne mein dikkat aayi: {e}")

st.subheader("📸 Choose Input Source")
run_webcam = st.checkbox("Turn on Webcam 🎥")

# Placeholder for image frames
FRAME_WINDOW = st.image([])

if run_webcam:
    # 0 is usually the built-in laptop webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Webcam open nahi ho paa raha hai. Please check permissions!")
    else:
        st.toast("Webcam started successfully! 🚀", icon="✅")
        
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame.")
                break
                
            # Run YOLOv8 object detection on the frame
            results = model(frame, stream=True, verbose=False)
            
            # Plot the boxes and labels on the frame
            for r in results:
                frame = r.plot()
                
            # Convert BGR (OpenCV format) to RGB (Streamlit format)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update the image placeholder continuously
            FRAME_WINDOW.image(frame_rgb)
            
        cap.release()
else:
    st.info("Webcam chalu karne ke liye upar diye gaye checkbox ko click karein.")