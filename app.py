import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import nltk
import string
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download stopwords (only runs first time)
nltk.download('stopwords')

# Load artifacts
model = tf.keras.models.load_model("spam_detector_model.keras")
tokenizer = joblib.load("tokenizer.pkl")

# Parameters (must match training)
max_len = 100  # ⚠️ adjust if your notebook used a different value

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Streamlit UI
st.title("📧 Spam Email Detector")
st.write("Enter an email message below to check if it's spam or not.")

user_input = st.text_area("Email Content")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        processed = preprocess_text(user_input)

        sequence = tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(sequence, maxlen=max_len)

        prediction = model.predict(padded)[0][0]

        # Because you used sigmoid
        if prediction > 0.5:
            st.error(f"🚨 Spam detected (confidence: {prediction:.2f})")
        else:
            st.success(f"✅ Not Spam (confidence: {1 - prediction:.2f})")
