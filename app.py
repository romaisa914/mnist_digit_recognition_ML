import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import joblib
from skimage.transform import resize

# Load the trained model
model = joblib.load("model.pkl")

st.title("MNIST Digit Recognition")
st.markdown("Draw a digit (0–9) below and click **Predict** to see the result.")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=200,
    height=200,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data

    if st.button("Predict"):
        # Convert RGBA to grayscale
        img_gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        
        # Resize to 8x8 like the original digits dataset
        img_resized = resize(img_gray, (8, 8), anti_aliasing=True)

        # Invert colors (white digit on black background)
        img_resized = 1.0 - img_resized

        # Scale pixel values to match training data (0–16)
        img_scaled = (img_resized * 16).astype(np.int32)

        # Flatten the image
        img_flattened = img_scaled.reshape(1, -1)

        # Make prediction
        prediction = model.predict(img_flattened)[0]

        st.subheader(f"Predicted Digit: {prediction}")
