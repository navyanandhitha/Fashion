# fashion_mnist_user_input.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
from PIL import Image, ImageOps

# Title
st.title("🧥 Fashion-MNIST Clothing Type Classifier ")


# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load dataset (for training)
@st.cache_data
def load_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)
    return train_images, train_labels, test_images, test_labels

train_images, train_labels, test_images, test_labels = load_data()

# Build model
@st.cache_resource
def build_and_train():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5, validation_split=0.2, verbose=0)
    return model

with st.spinner("Training model... Please wait ⏳"):
    model = build_and_train()

# Upload image
uploaded_file = st.file_uploader("Upload a clothing image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize and normalize
    image = ImageOps.invert(image)  # Invert colors (Fashion-MNIST is white on black)
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    confidence = 100 * np.max(prediction)

    # Force women's wear types to Dress
    predicted_class = class_names[predicted_label]
    if predicted_class in ["Shirt", "T-shirt/top"]:
        predicted_class = "Dress (includes frocks, chudidhars, gowns)"

    # Output
    st.subheader(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

    # Show confidence scores
    st.bar_chart(prediction[0])
