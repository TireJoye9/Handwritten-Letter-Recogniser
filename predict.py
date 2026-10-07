import os

# Keep TensorFlow quiet — these need to be set before tf is imported
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds


# Load the model we trained earlier
model = tf.keras.models.load_model('best_model.keras')

# Walk through UpperCaseLetters/Letter1.png, Letter2.png, ... until one is missing
image_number = 1
while os.path.isfile(f"UpperCaseLetters/Letter{image_number}.png"):
    try:
        # Load the image as a single-channel grayscale array
        img = cv2.imread(f"UpperCaseLetters/Letter{image_number}.png", cv2.IMREAD_GRAYSCALE)

        # Shrink to 28x28, the size the model was trained on
        img = cv2.resize(img, (28, 28))

        # Flip black-on-white (what MS Paint gives us) to white-on-black
        # (what EMNIST looks like)
        img = np.invert(img)

        # Scale pixel values from 0-255 down to 0-1
        img = img.astype(np.float32) / 255.0

        # The model expects a batch dimension and a channel dimension,
        # so reshape from (28, 28) to (1, 28, 28, 1)
        img = img.reshape(1, 28, 28, 1)

        # Ask the model what it thinks this letter is
        prediction = model.predict(img, verbose=0)

        # prediction is 27 probabilities — take the highest one
        label = int(np.argmax(prediction))

        # Label 1 = A, 2 = B, ..., 26 = Z. Label 0 is unused, so guard it.
        letter = chr(label + 64) if label > 0 else "?"
        confidence = prediction[0][label]
        print(f"Letter {image_number}: {letter} ({confidence:.1%})")

        # Show the preprocessed image so we can see what the model actually saw.
        # If the prediction looks wrong, this is the first thing to check.
        plt.imshow(img[0].reshape(28, 28), cmap=plt.cm.binary)
        plt.title(f"Predicted: {letter}")
        plt.show()

    except Exception as e:
        # Print the real error instead of swallowing it
        print(f"Error on {image_number}: {e}")
    finally:
        image_number += 1