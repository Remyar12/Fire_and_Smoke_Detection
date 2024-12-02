import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import base64

# Load the pre-trained model
def load_model():
    return tf.keras.models.load_model(r"C:\Users\remya\OneDrive\Desktop\fire_and_smoke_original\result\cnn_fire_and_smoke.keras")

model = load_model()

# Preprocess the uploaded image
def preprocess_image(image):
    # Resize the image to 128x128
    image = image.resize((128, 128))
    
    # Convert to RGB if the image is not already in RGB mode
    image = image.convert("RGB")
    
    # Convert to NumPy array
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    
    # Ensure image has the correct shape (128, 128, 3)
    if image.shape != (128, 128, 3):
        raise ValueError("Image preprocessing failed. The image must have shape (128, 128, 3).")
    
    # Add a batch dimension (1, 128, 128, 3)
    image = np.expand_dims(image, axis=0)
    
    return image

def make_prediction(img, model):
    # Preprocess the image
    img = Image.fromarray(img)
    img = img.resize((128, 128))  # Resize the image to the expected input size
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)

    # Normalize the image if required
    input_img = tf.keras.utils.normalize(input_img, axis=1)

    # Make prediction
    res = model.predict(input_img)

    # Print the output probabilities to see the model's decision
    print("Model output probabilities:", res)

    # Get the class with the highest probability
    predicted_class = np.argmax(res, axis=1)[0]

    # Print the predicted class label
    if predicted_class == 0:
        print("Class 0: No Fire and Smoke")
    elif predicted_class == 1:
        print("Class 1: Low Intensity Fire and Smoke")
    elif predicted_class == 2:
        print("Class 2: High Intensity Fire and Smoke")
    else:
        print("Unknown class")

    return predicted_class

# Streamlit UI
st.title("Fire and Smoke Detection")
st.write("Upload an image to check for fire and smoke detection.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Load and display the image
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image.", use_column_width=True)

    # Convert image to numpy array
    img_array = np.array(img)

    # Call prediction function
    predicted_class = make_prediction(img_array, model)

    # Display the results
    if predicted_class == 0:
        st.write("Class 0: No Fire and Smoke")
    elif predicted_class == 1:
        st.write("Class 1: Low Intensity Fire and Smoke")
    elif predicted_class == 2:
        st.write("Class 2: High Intensity Fire and Smoke")
    else:
        st.write("Unknown class")

    # Display alert if fire or smoke is detected
    if predicted_class != 0:
        st.error("🚨 ALERT! Fire and Smoke Detected! 🚨")

        # Encode the danger image to base64 for embedding
        def get_base64_encoded_image(image_path):
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            return encoded_image

        danger_icon_path = r"C:\Users\remya\OneDrive\Desktop\fire_and_smoke_original\danger_image.png"

        # Encode the danger image to base64
        encoded_image = get_base64_encoded_image(danger_icon_path)

        # Display the danger image in the center using HTML and CSS
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; margin-top: 20px;">
                <img src="data:image/png;base64,{encoded_image}" alt="ALARM!" style="max-width: 300px;"/>
            </div>
            """,
            unsafe_allow_html=True,
        )
