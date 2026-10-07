# Handwritten Letter Recognition

A convolutional neural network that recognizes handwritten uppercase letters (A–Z). Trained on the EMNIST Letters dataset using TensorFlow and Keras.

## Preview
<img width="1722" height="686" alt="Screenshot 2026-10-07 012841" src="https://github.com/user-attachments/assets/d91fde07-0903-4728-a905-58cf9e0b0ee8" />

## What it does

The model takes a 28×28 grayscale image of a handwritten uppercase letter and predicts which letter it is. There are 27 output classes: indices 1–26 map to A–Z, and index 0 is unused.

## Results

| Model | Dataset | Accuracy |
|---|---|---|
| Dense network (baseline) | EMNIST Letters test set | ~80% |
| CNN | EMNIST Letters test set | ~93% |
| CNN | My own handwritten letters | ~77% (20/26) |

The first version of this project used a plain fully-connected network. It flattened each 28×28 image into a 784-length vector and fed it through two hidden layers. That model capped out around 80% no matter how many epochs I trained for, which is the expected ceiling for a dense architecture on image data. Flattening destroys spatial information, so the network can only learn per-pixel intensity patterns, not shapes or strokes.

Switching to a CNN with two convolutional blocks pushed the same task to ~93%. The convolutional layers preserve the 2D layout of the image, so the model can learn edges in the first block, then combine them into curves and corners in the second. That spatial awareness is worth roughly 13 percentage points on this dataset, and it does it with fewer parameters than the dense version.

The gap between the ~93% on the EMNIST test set and the ~77% on my own handwriting is domain shift. EMNIST was collected from a specific group of writers, and my handwriting doesn't match their style exactly. On my own letters, the model's errors are concentrated in visually similar pairs at 28×28 resolution: D/B, N/M, O/Q, S/G, V/U, X/Y. When the model gets one of these wrong, it usually reports low confidence (around 50%), which is the expected behavior.

## Requirements

```
tensorflow>=2.15
tensorflow-datasets>=4.9
opencv-python>=4.8
numpy>=1.24
matplotlib>=3.7
```

Install with:

```
pip install -r requirements.txt
```

## Training

```
python train.py
```

The first run downloads the EMNIST Letters dataset (~535 MB) to your local TensorFlow Datasets cache. Subsequent runs load from cache instantly.

Training uses EarlyStopping and ModelCheckpoint callbacks. EarlyStopping halts training if validation loss hasn't improved for 5 consecutive epochs, and ModelCheckpoint saves the best weights to `best_model.keras` whenever validation loss improves. This means you can stop training at any time without losing your best result.

On CPU, a full training run takes around 20 to 40 minutes depending on how early EarlyStopping fires.

## Prediction

1. Draw uppercase letters in MS Paint or any image editor, using a black letter on a white background. Make the letter fill most of the canvas.
2. Save them in a folder called `UpperCaseLetters/` as `Letter1.png`, `Letter2.png`, and so on.
3. Run:

```
python predict.py
```

The script walks through every `LetterN.png` file in order, preprocesses each one to match the training pipeline, and prints the predicted letter with confidence. It also displays what the model actually sees after preprocessing, which is useful for debugging when a prediction looks wrong.

Example output:

```
Letter 1: A (98.7%)
Letter 2: B (99.9%)
Letter 3: C (55.2%)
Letter 4: B (57.9%)
Letter 5: E (99.8%)
```

## Model architecture

The model is a CNN with two convolutional blocks followed by a classifier head.

```
Input (28×28×1)
  Conv2D(32, 3×3, relu, same padding) → BatchNorm → MaxPool(2×2)
  Conv2D(64, 3×3, relu, same padding) → BatchNorm → MaxPool(2×2)
  Flatten
  Dense(128, relu) → Dropout(0.3)
  Dense(27, softmax)
```

The first conv block learns low-level features like edges and strokes. The second block combines those into higher-level shapes like curves and corners. The classifier head flattens the feature maps and produces a probability distribution over the 27 output classes.

## Notes

**Uppercase only.** The model was trained on `emnist/letters`, which contains only uppercase A–Z. It does not recognize lowercase letters or digits. There are no output neurons for those classes, so the model will always produce one of the 26 uppercase letters regardless of what you feed it.

**Input format matters.** Images should be black letters on a white background, drawn large enough to fill most of the canvas. The prediction script handles the conversion to EMNIST's format, but if you feed the model an already-preprocessed image, you'll double-process it.

**The transpose step is essential.** EMNIST images are stored rotated relative to normal images. Skipping the transpose in either training or prediction drops accuracy sharply, usually to around 60%. If predictions look consistently wrong, check the preprocessing visualization first.

## Project structure

```
NeuralNetwork/
├── train.py                  Training script
├── predict.py                Prediction script
├── best_model.keras          Trained model weights
├── requirements.txt
├── README.md
├── .gitignore
└── UpperCaseLetters/         Input images for prediction
    ├── Letter1.png
    └── ...
```

## Stack

Python, TensorFlow, Keras, TensorFlow Datasets, OpenCV, NumPy, Matplotlib.
