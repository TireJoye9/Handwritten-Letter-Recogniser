import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

""""
# load emnist letters
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def preprocess(image, label):
    # EMNIST images are stored transposed relative to MNIST
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

BATCH_SIZE = 64
ds_train = ds_train.map(preprocess).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_test  = ds_test.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# CNN Creation
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    # learn simple features (edges, strokes)
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # learn combinations (curves, corners)
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Classifier head
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(27, activation='softmax'),   # 27 outputs: labels 1-26
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# saves best model to disk whenever val_loss improves
checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

history = model.fit(
    ds_train,
    epochs=30,
    validation_data=ds_test,
    callbacks=[early_stop, checkpoint]
)

model.save('handwritten_letters_cnn.keras')
"""
model = tf.keras.models.load_model('best_model.keras')
#loss, accuracy = model.evaluate(ds_test)
#print("Loss:", loss)
#print("Accuracy:", accuracy)

image_number = 1
while os.path.isfile(f"UpperCaseLetters/Letter{image_number}.png"):
    try:
        #Load as grayscale
        img = cv2.imread(f"UpperCaseLetters/Letter{image_number}.png", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = np.invert(img)
        img = img.astype(np.float32) / 255.0
        #Add batch + channel dims → (1, 28, 28, 1)
        img = img.reshape(1, 28, 28, 1)

        prediction = model.predict(img, verbose=0)
        label = int(np.argmax(prediction))
        letter = chr(label + 64) if label > 0 else "?"
        confidence = prediction[0][label]
        print(f"Letter {image_number}: {letter} ({confidence:.1%})")

        #Show what the model actually sees
        plt.imshow(img[0].reshape(28, 28), cmap=plt.cm.binary)
        plt.title(f"Predicted: {letter}")
        plt.show()

    except Exception as e:
        print(f"Error on {image_number}: {e}")
    finally:
        image_number += 1
