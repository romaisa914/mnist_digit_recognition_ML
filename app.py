import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
from skimage.transform import resize
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("MNIST Digit Recognizer")
st.markdown("Draw a digit (0-9) below:")

# Create canvas
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype("uint8"))
        img = img.resize((8, 8)).convert("L")  # Resize to 8x8
        img_array = np.array(img)
        img_array = 16 - (img_array / 16).astype("int")  # Normalize to 0-16 scale
        img_flattened = img_array.flatten().reshape(1, -1)
        prediction = model.predict(img_flattened)[0]
        st.success(f"Predicted Digit: {prediction}")
    else:
        st.warning("Please draw a digit before predicting.")
