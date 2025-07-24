import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import joblib
from PIL import Image
from skimage.transform import resize

# Load trained model
model = joblib.load("model.pkl")

st.title("🖌️ MNIST Digit Recognizer (Draw Your Digit)")
st.write("Draw a digit (0-9) below and click Predict!")

# Create a canvas
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

if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert to grayscale and resize to 8x8 like MNIST
        img = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype('uint8')).convert('L')
        img_resized = resize(np.array(img), (8, 8), anti_aliasing=True)
        img_rescaled = (16 - (img_resized / 255.0 * 16)).reshape(1, -1)  # Invert to match MNIST
        prediction = model.predict(img_rescaled)[0]
        st.success(f"🎯 Predicted Digit: **{prediction}**")
    else:
        st.warning("Please draw something first.")
