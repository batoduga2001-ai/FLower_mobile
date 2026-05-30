import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Flower Classifier", layout="centered")

st.title("🌸 Flower Image Classifier")
st.write("Upload an image and the model will predict the flower type.")

# ======================
# LOAD MODEL (IMPORTANT)
# ======================
@st.cache_resource
def load_my_model():
    model = load_model("flower_mobilenet.keras")  # make sure file exists in repo
    return model

model = load_my_model()

# Class names (must match training order)
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# ======================
# IMAGE UPLOAD
# ======================
uploaded_file = st.file_uploader("Upload a flower image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ======================
    # PREPROCESS IMAGE
    # ======================
    img = image.resize((224, 224))  # adjust if your model uses different size
    img_array = np.array(img)

    # normalize
    img_array = img_array / 255.0

    # add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # ======================
    # PREDICT
    # ======================
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    confidence = np.max(predictions) * 100

    # ======================
    # RESULT
    # ======================
    st.subheader("Prediction Result")
    st.success(f"🌼 Flower: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")