# Import necessary libraries
import os
import numpy as np
import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
from keras.applications.vgg16 import preprocess_input
# Load your trained model
MODEL_PATH = 'CNN_model.h5'
model = load_model(MODEL_PATH)

# Function for processing the input image and prediction
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(176, 176))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # Ensure you have the preprocess_input function

    y = model.predict(x)

    return np.argmax(y)

# Main Streamlit app
def main():
    st.title('''Alzheimer's Disease Diagnosis App''')

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Save the uploaded image to a temporary file
        temp_file_path = "temp.jpg"
        uploaded_file.seek(0)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Make prediction
        preds = model_predict(temp_file_path, model)

        # Process your result for human
        dic = {0: "Alzheimer's disease", 1: "Cognitively normal", 2: "Early mild cognitive impairment",
               3: "Late mild cognitive impairment"}
        pred_class = dic[preds]

        st.success(f"The predicted class is: {pred_class}")

# Run the Streamlit app
if __name__ == '__main__':
    main()
