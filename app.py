import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageOps

# Load model
model = joblib.load("model.pkl")

st.title("MNIST Digit Recognizer")
uploaded_file = st.file_uploader("Upload a digit image (black on white)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = ImageOps.invert(image)
    image = image.resize((8, 8))  # Resize to 8x8
    image = np.array(image)
    image = 16 * image / 255.0
    image = image.reshape(1, -1)

    prediction = model.predict(image)[0]
    st.success(f"Predicted Digit: {prediction}")
