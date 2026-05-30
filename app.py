import streamlit as st
import numpy as np
from PIL import Image
import pickle

# Load model (NO tensorflow)
model = pickle.load(open("model.pkl", "rb"))

class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

st.title("🌸 Flower Classifier")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    st.write("⚠ Model requires preprocessing pipeline (you must match training)")

    st.success("Prediction system ready (fix model format needed)")