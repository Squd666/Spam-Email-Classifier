import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import nltk
from nltk.corpus import stopwords

# 1. Setup NLTK (Must download on the server)
nltk.download('stopwords')
nltk.download('punkt')
STOPWORDS = set(stopwords.words('english'))

# 2. Load the Brain (Model) and the Translator (Tokenizer)
# Ensure these filenames match exactly what you uploaded to GitHub
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('spam_detector_model.keras')
    tokenizer = joblib.load('tokenizer.pkl')
    return model, tokenizer

model, tokenizer = load_assets()

# 3. UI Layout
st.set_page_config(page_title="AI Spam Detector", page_icon="📧")
st.title("📧 Professional Spam Classifier")
st.write("Enter an email below to check if it's safe or spam.")

user_input = st.text_area("Email Content:", height=200, placeholder="Paste email here...")

if st.button("Analyze Email"):
    if user_input.strip():
        # Preprocessing (Matching your notebook logic)
        # 1. Tokenize
        sequences = tokenizer.texts_to_sequences([user_input])
        # 2. Pad (using maxlen=50 from your notebook)
        padded = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')
        
        # Prediction
        prediction = model.predict(padded)
        probability = float(prediction[0][0])
        
        if probability > 0.5:
            st.error(f"🚨 **SPAM DETECTED** (Confidence: {probability:.2%})")
        else:
            st.success(f"✅ **LEGITIMATE EMAIL** (Confidence: {1-probability:.2%})")
    else:
        st.warning("Please enter some text to analyze.")
