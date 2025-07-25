import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageOps

# Load the trained model
model = joblib.load("model.pkl")

st.title("MNIST Digit Recognizer")
st.write("Upload an image of a digit (0-9) written in black on white background.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = ImageOps.invert(image)
    image = image.resize((8, 8))  # Downscale to 8x8 to match sklearn digits
    image = np.array(image)
    image = 16 * image / 255.0  # Scale to match dataset
    flat_data = image.reshape(1, -1)

    # Predict
    prediction = model.predict(flat_data)[0]
    st.write(f"### Predicted Digit: {prediction}")
