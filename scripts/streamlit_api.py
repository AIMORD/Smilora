import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import google.generativeai as genai
from collections import Counter

# --- Page Configuration ---
st.set_page_config(page_title="Smilora Dental AI Assistant", page_icon="🦷", layout="wide")

# --- Model Loading ---
@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)

# --- Gemini Report Generation ---
def generate_gemini_report(api_key, image, detections):
    """Generates a patient-friendly report using the Gemini API."""
    genai.configure(api_key=api_key)
    
    # Create a summary of the detections for the prompt
    detection_summary = ", ".join([f"{count} {label}" for label, count in detections.items()])
    
    # --- The New, More Detailed Prompt ---
    prompt = f"""
    You are an AI dental health educator. Your goal is to analyze the provided image and the pre-analyzed findings to generate an educational report for a patient. The language must be simple, clear, and reassuring.

    The computer vision model detected the following potential issues: **{detection_summary}**.

    Based on the image and these findings, generate a report with the following sections:

    **1. AI Analysis Summary:**
    Start with a friendly opening and list the items detected by the AI.

    **2. Understanding Your Scan:**
    For each unique issue detected (e.g., 'Cavity', 'Crack'), create a small section. In each section, explain the following in one simple sentence each:
    * **What it is:** A simple definition (e.g., "A cavity is a small hole that can form in a tooth.").
    * **Common Causes:** Briefly mention one or two common causes (e.g., "They are often caused by sugary foods and drinks without proper brushing.").
    * **Why a Check-up is Important:** Explain why it's important to have a dentist look at it (e.g., "It's important to have a dentist check a potential cavity to prevent it from getting larger or causing discomfort.").

    **3. General Recommendations:**
    * **Next Step:** Conclude with a clear and strong recommendation to schedule a visit with a qualified dentist for a professional examination and formal diagnosis.
    * **General Tips:** Provide 2-3 general, actionable tips for good oral hygiene (e.g., brushing twice a day, flossing, regular check-ups).

    **CRITICAL INSTRUCTIONS:**
    - DO NOT provide a medical diagnosis.
    - DO NOT use complex or alarming medical jargon.
    - DO NOT suggest specific medical treatments.
    - The entire report should be encouraging and focused on promoting a visit to a professional dentist.
    """
    
    # Load the Gemini model
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    # Generate the content
    response = model.generate_content([prompt, image])
    return response.text

# --- Main Application ---
st.title("🦷 AI Dental Health Assistant")

# --- Setup ---
MODEL_PATH = 'results/best.pt'
model = load_yolo_model(MODEL_PATH)

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("Gemini API key not found. Please create a .streamlit/secrets.toml file with your key.")
    GEMINI_API_KEY = None

if model and GEMINI_API_KEY:
    # --- Image Upload ---
    uploaded_file = st.file_uploader("Upload an image of your teeth...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        
        st.write("### Analysis Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(pil_image, caption="Uploaded Image", use_column_width=True)
            
        with col2:
            with st.spinner("Running analysis... This may take a moment."):
                # Run YOLO model
                results = model(pil_image, verbose=False)
                annotated_image = results[0].plot()[..., ::-1] # Plot and convert to RGB
                
                # Get a list of detected class names
                detected_classes = [model.names[int(c)] for c in results[0].boxes.cls]
                # Filter out 'Tooth' class for the report
                disease_detections = Counter([cls for cls in detected_classes if cls != 'Tooth'])

                st.image(annotated_image, caption="AI Detection", use_column_width=True)
        
        st.write("---")
        st.write("### AI Generated Report")
        
        if not disease_detections:
            st.success("The AI did not detect any of the specific disease classes it was trained on. However, this is not a diagnosis. A check-up with a dentist is always recommended for a complete evaluation.")
        else:
            with st.spinner("Generating your educational report with Gemini..."):
                # Generate and display the report from Gemini
                report = generate_gemini_report(GEMINI_API_KEY, pil_image, disease_detections)
                st.markdown(report)
                
        st.warning("⚠️ **Disclaimer:** This is an AI-generated analysis and is for informational purposes only. It is not a substitute for a professional medical diagnosis. Please consult a qualified dentist for any health concerns.")