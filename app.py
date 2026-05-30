from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Load model
model = load_model("model.keras")

class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


def preprocess_image(image):
    img = Image.open(image).resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files["image"]
        img = preprocess_image(file)

        pred = model.predict(img)
        class_index = np.argmax(pred)
        prediction = class_names[class_index]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)