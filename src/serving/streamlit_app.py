import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
import json

# Page config
st.set_page_config(page_title="MNIST Digit Classifier", page_icon="🔢")

# Title
st.title("🔢 MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9) for classification")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('L')  #Convert to grayscale
    st.image(image, caption='Uploaded Image', use_container_width=True)    
    # Preprocess image
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    
    # Prepare for API call
    pixel_list = image_array.flatten().tolist()
    
    # Make prediction via API
    if st.button('Classify Digit'):
        try:
            response = requests.post(
                "http://localhost:8001/predict",
                json={"image": pixel_list}
            )
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"**Prediction:** {result['prediction']}")
                st.info(f"**Confidence:** {result['confidence']:.2%}")
                
                # Show probabilities
                st.write("**Class Probabilities:**")
                for i, prob in enumerate(result['probabilities']):
                    st.write(f"Digit {i}: {prob:.2%}")
            else:
                st.error("Prediction failed. Please check if the API server is running.")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")