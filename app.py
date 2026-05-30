import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model("flower_model.keras")

class_names = ['daisy','dandelion','rose','sunflower','tulip']

st.title("🌸 Flower Classifier")

uploaded_file = st.file_uploader("Upload an image")

if uploaded_file:
    img = Image.open(uploaded_file).resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    result = class_names[np.argmax(pred)]

    st.success(f"Prediction: {result}")