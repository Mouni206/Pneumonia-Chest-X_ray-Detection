import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from util import classify

# Load the model
model_path = r'C:\Users\rmoun\OneDrive\Documents\pythonProject_pn\model\pneumonia.h5'
try:
    model = load_model(model_path)
    st.write(f"Model loaded successfully from: {model_path}")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Define class names
class_names = ['Normal', 'Pneumonia']

# App Title
st.title('Pneumonia Classification')
st.header('Please upload a chest X-ray image')

# File Upload
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Predict and Display Result
if file is not None:
    try:
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)

        class_name, conf_score = classify(image, model, class_names)

        st.write(f"## Prediction: {class_name}")
        st.write(f"### Confidence Score: {conf_score * 100:.2f}%")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
