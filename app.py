import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

# ------------------------------------------------
# Load the trained model (includes TextVectorization)
# ------------------------------------------------
@st.cache_resource
def load_toxicity_model():
    model = load_model("toxicity.h5")
    return model

model = load_toxicity_model()

# ------------------------------------------------
# Streamlit UI
# ------------------------------------------------
st.set_page_config(page_title="Negative Sentiment Detector", page_icon="😠", layout="centered")

st.title("😠 Negative Sentiment Prediction")
st.write("Enter a sentence to check if it expresses **negative sentiment**.")

# Input box
user_input = st.text_area("✍️ Type your text here:", placeholder="I really dislike this product...")

# Predict button
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        # Preprocess and predict
        prediction = model.predict([user_input])[0][0]

        # Show result
        st.subheader("🔮 Prediction Result")
        if prediction > 0.5:
            st.error(f"Negative Sentiment Detected 😡 (Confidence: {prediction:.2f})")
        else:
            st.success(f"Not Negative 🙂 (Confidence: {1 - prediction:.2f})")
