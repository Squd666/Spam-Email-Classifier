import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

# Load the modern model format and the tokenizer
model = tf.keras.models.load_model('spam_detector_model.keras')
tokenizer = joblib.load('tokenizer.pkl')

st.set_page_config(page_title="Email Spam Detector", page_icon="📧")
st.title("📧 NLP Spam Classifier")

user_input = st.text_area("Paste the email content below:", height=200)

if st.button("Predict"):
    if user_input.strip():
        # Preprocess using the exact same steps as your notebook
        # maxlen=50 based on your notebook's training configuration
        sequences = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')
        
        prediction = model.predict(padded)
        score = prediction[0][0]
        
        if score > 0.5:
            st.error(f"🚨 This is likely SPAM! (Probability: {score:.2f})")
        else:
            st.success(f"✅ This is a legitimate email. (Probability: {score:.2f})")
    else:
        st.warning("Please enter some text.")