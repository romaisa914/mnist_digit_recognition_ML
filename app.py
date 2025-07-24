import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import joblib
from PIL import Image
from skimage.transform import resize

# Load trained model
model = joblib.load("model.pkl")

st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")
st.title("🖌️ MNIST Digit Recognizer")
st.write("Draw a digit (0-9) below and click **Predict**!")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Prediction logic
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert to grayscale (remove alpha) and resize to 8x8
        img = Image.fromarray((canvas_result.image_data[:, :, :3]).astype("uint8")).convert("L")
        img_resized = resize(np.array(img), (8, 8), anti_aliasing=True)

        # Invert colors and scale to 0–16 like MNIST
        img_rescaled = 16 - (img_resized / 255.0 * 16)

        # Flatten and clean data
        img_flattened = img_rescaled.reshape(1, -1)
        img_flattened = np.nan_to_num(img_flattened).astype("float64")

        # Predict
        prediction = model.predict(img_flattened)[0]
        st.success(f"🎯 Predicted Digit: **{prediction}**")
    else:
        st.warning("⚠️ Please draw a digit first!")
