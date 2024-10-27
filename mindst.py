import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("cnn_mnist_model.h5")

st.title("MNIST Digit Classifier")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Preprocess the image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)  # Reshape for the model
    
    # Display the image
    st.image(image.reshape(28, 28), width=150, caption="Uploaded Image")
    
    # Make prediction
    prediction = model.predict(image)
    st.write(f"Predicted Digit: {np.argmax(prediction)}")
