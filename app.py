import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os

@st.cache_resource
def load_toxicity_model():
    # Make sure the model file exists
    model_path = "toxicity.h5"
    
    if os.path.exists(model_path):
        model = load_model(model_path)
        return model
    else:
        st.error(f"❌ Model file not found at {model_path}")
        return None

model = load_toxicity_model()

if model is not None:
    st.title("😠 Negative Sentiment Prediction")
    user_input = st.text_area("✍️ Type your text here:", placeholder="I really dislike this product...")

    if st.button("🔮 Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text before predicting.")
        else:
            with st.spinner("Analyzing sentiment..."):
                prediction = model.predict([user_input])[0][0]
            
            confidence = float(prediction if prediction > 0.5 else 1 - prediction)

            if prediction > 0.5:
                st.markdown(f"😡 Negative Sentiment Detected (Confidence: {confidence:.2f})")
            else:
                st.markdown(f"🙂 Not Negative (Confidence: {confidence:.2f})")
else:
    st.stop()  # Stop execution if model failed to load
