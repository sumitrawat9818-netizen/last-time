import streamlit as st
import pickle
import os
import sys
from google import genai
from clean_text import clean_text 

# --- 1. LLM Setup (The Therapy Brain) ---
try:
    # The client automatically looks for "GEMINI_API_KEY" in Streamlit Secrets
    client = genai.Client()
except Exception:
    client = None

def generate_therapy_response(user_input: str, predicted_intent: str) -> str:
    """
    Generates an empathetic response using the Gemini API.
    """
    if not client:
        return "Error: API Key missing. Please set GEMINI_API_KEY in Secrets."
    
    PROMPT = f"""
    You are a compassionate AI mental health assistant named 'Clarity'.
    The user's intent is classified as: {predicted_intent}.
    The user said: "{user_input}"
    
    Provide a supportive, structured response (3-4 sentences):
    1. Validate their feeling (e.g., "It sounds like you are feeling...").
    2. Offer one simple, actionable coping tip (e.g., breathing, walking).
    3. Gently suggest professional help if the topic is serious.
    """
    
    try:
        # Using the 2.5 Flash model for speed and quality
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=PROMPT
        )
        return response.text
    except Exception as e:
        return f"Connection Error: {e}"

# --- 2. Load Model Assets ---
try:
    # Load models from the ROOT directory
    with open('intent_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    st.error("Critical Error: .pkl files are missing from the repository root.")
    st.stop()

# --- 3. UI Layout ---
st.set_page_config(page_title="Clarity: AI Therapy", layout="centered")
st.title("🤖 Clarity: Mental Health Assistant")
st.markdown("A Hybrid AI System: **Machine Learning** (Intent) + **Generative AI** (Response)")

# Chat Input
user_input = st.text_area("How are you feeling right now?", height=100)

if st.button("Get Support"):
    if user_input:
        # A. ML Prediction Step
        st.info("Analyzing sentiment...")
        
        # Clean and Vectorize
        cleaned_text = clean_text(user_input)
        input_vec = vectorizer.transform([cleaned_text])
        
        # Predict Intent
        predicted_intent = model.predict(input_vec)[0]
        
        # Show the classification result (Great for your project demo!)
        st.sidebar.success(f"Detected Intent: **{predicted_intent.upper()}**")
        
        # B. Generative AI Step
        with st.spinner("Generating compassionate response..."):
            ai_response = generate_therapy_response(user_input, predicted_intent)
            
            st.subheader("Clarity's Response:")
            st.write(ai_response)
    else:
        st.warning("Please type something first.")