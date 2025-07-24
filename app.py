import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("model.pkl")

st.title("🧠 MNIST Digit Recognizer")
st.write("Enter 64 pixel values for an 8x8 image (values: 0 to 16):")

# Create 8x8 grid inputs
pixels = []
for row in range(8):
    cols = st.columns(8)
    for col in cols:
        val = col.number_input(" ", min_value=0, max_value=16, step=1, key=f"{row}{col}")
        pixels.append(val)

# Predict
if st.button("Predict Digit"):
    prediction = model.predict([pixels])[0]
    st.success(f"🟢 Predicted Digit: **{prediction}**")