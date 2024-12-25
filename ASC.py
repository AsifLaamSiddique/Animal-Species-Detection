#   python -m venv myenv
#   myenv\Scripts\activate
#   python ASC.py


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
import random
import pathlib

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Paths to dataset directories
train_path = "H:\\OpenCV\\Animal Species Classification\\Animal Species Classification Dataset\\Training Data"
val_path = "H:\\OpenCV\\Animal Species Classification\\Animal Species Classification Dataset\\Validation Data"
test_path = "H:\\OpenCV\\Animal Species Classification\\Animal Species Classification Dataset\\Testing Data"

# Create dataset directories
train_data_dir = pathlib.Path(train_path)
val_data_dir = pathlib.Path(val_path)
test_data_dir = pathlib.Path(test_path)

# Get class names
class_names = np.array(sorted([item.name for item in train_data_dir.glob('*') if item.is_dir()]))

# Define image size and batch size
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32

# Rescaling and data augmentation layers
rescale_data = tf.keras.Sequential([
    layers.Rescaling(1./255)
])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip(mode="horizontal"),
    layers.RandomRotation(0.2)
])

# Function to preprocess datasets
def prepare_dataset(directory):
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

# Load datasets
train_ds = prepare_dataset(train_path)
val_ds = prepare_dataset(val_path)
test_ds = prepare_dataset(test_path)

# Apply rescaling and augmentation to training data
train_ds = train_ds.map(lambda x, y: (rescale_data(data_augmentation(x, training=True)), y))
val_ds = val_ds.map(lambda x, y: (rescale_data(x), y))
test_ds = test_ds.map(lambda x, y: (rescale_data(x), y))

# Prefetch datasets for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# Define the model
model = tf.keras.models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMAGE_SIZE + (3,)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

# Compile the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

# Train the model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)

# Save the model
model.save('animal_species_model.keras')

# Load the model
loaded_model = tf.keras.models.load_model('animal_species_model.keras')

# Function to preprocess and predict on a random image
def predict_random_image(directory):
    random_class = random.choice(os.listdir(directory))
    random_image = random.choice(os.listdir(os.path.join(directory, random_class)))
    random_image_path = os.path.join(directory, random_class, random_image)

    img = load_img(random_image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = loaded_model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    print(f"Random Image Path: {random_image_path}")
    print(f"Predicted Class: {predicted_class}")

# Predict on a random image from the test dataset
predict_random_image(test_path)
