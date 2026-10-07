import os

# Keep TensorFlow quiet — these must be set before tf is imported
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Load the EMNIST Letters dataset (uppercase A-Z, 88k train + 14.8k test images)
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def preprocess(image, label):
    # EMNIST images are stored transposed (rotated) relative to MNIST,
    # so we flip the first two axes to get them upright
    image = tf.transpose(image, perm=[1, 0, 2])
    # Scale pixel values from 0-255 down to 0-1 — models train better this way
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

BATCH_SIZE = 64
# Training data gets shuffled so the model doesn't see letters in order;
# test data doesn't need shuffling since we're just measuring accuracy
ds_train = ds_train.map(preprocess).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_test  = ds_test.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Build the CNN — two convolutional blocks followed by a classifier head
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    # First conv block — learns simple features like edges and strokes
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Second conv block — combines those features into curves and corners
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Classifier head — flattens the feature maps and makes the final decision
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),   # randomly drop 30% of neurons to prevent overfitting
    tf.keras.layers.Dense(27, activation='softmax'),   # 27 outputs: labels 1-26 map to A-Z, 0 is unused
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',   # standard loss for integer-labeled classification
    metrics=['accuracy']
)

# Stop training early if validation loss hasn't improved for 5 epochs,
# and roll back to the best weights seen
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Save the model to disk whenever val_loss improves — makes Ctrl+C safe
checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Train — EarlyStopping will halt before 30 epochs if the model plateaus (halted at 7, took way to long - 92/3 %)
history = model.fit(
    ds_train,
    epochs=30,
    validation_data=ds_test,
    callbacks=[early_stop, checkpoint]
)

# Save the final model as well (in case we want the last-epoch weights too)
model.save('handwritten_letters_cnn.keras')

# Reload the best checkpoint and measure its true accuracy on the test set
model = tf.keras.models.load_model('best_model.keras')
loss, accuracy = model.evaluate(ds_test)
print("Loss:", loss)
print("Accuracy:", accuracy)