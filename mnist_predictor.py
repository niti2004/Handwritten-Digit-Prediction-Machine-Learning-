import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os
import cv2

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Reshape and normalize the data
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create a CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_val, y_val), 
                    verbose=2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.2f}')

# Function to preprocess uploaded images
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img = np.array(img)
    img = img > 128
    img = img.astype('float32') / 255
    img = img.reshape(1, 28, 28, 1)
    return img

# Function to predict uploaded images
def predict_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    return predicted_label

# Paths to the uploaded images
uploaded_image_paths = [
    r"C:\Users\agnih\Downloads\ml project\2.png",
    r"C:\Users\agnih\Downloads\ml project\1.png"
]

# Predict and visualize each uploaded image
for image_path in uploaded_image_paths:
    if os.path.exists(image_path):
        predicted_label = predict_image(image_path)
        print(f'Predicted label for {os.path.basename(image_path)}: {predicted_label}')
        
        # Visualize the uploaded image and prediction
        uploaded_img = Image.open(image_path).convert('L')
        plt.imshow(uploaded_img, cmap='gray')
        plt.title(f'Predicted: {predicted_label}')
        plt.xticks([])
        plt.yticks([])
        plt.show()
    else:
        print(f'File not found: {image_path}')