import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Flower Classifier", page_icon="🌸")

st.title("🌸 Flower Mobile Classifier")
st.write("Upload an image and the model will predict the flower type.")

# -------------------------
# LOAD MODEL (FIXED)
# -------------------------
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model("flower_mobile.keras")
    return model

model = load_my_model()

# -------------------------
# CLASS LABELS
# -------------------------
class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# -------------------------
# IMAGE UPLOAD
# -------------------------
uploaded_file = st.file_uploader("Upload Flower Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -------------------------
    # PREPROCESS IMAGE
    # -------------------------
    img = image.resize((224, 224))
    img_array = np.array(img)

    # normalize
    img_array = img_array / 255.0

    # expand dims for model input
    img_array = np.expand_dims(img_array, axis=0)

    # -------------------------
    # PREDICTION
    # -------------------------
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.subheader("Prediction:")
    st.success(f"🌸 {predicted_class}")