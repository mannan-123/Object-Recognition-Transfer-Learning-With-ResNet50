# CIFAR-10 Image Classification with Neural Networks

## Overview

This project focuses on classifying CIFAR-10 images using deep learning models, including a simple feedforward neural network and ResNet50. The dataset is downloaded from Kaggle, preprocessed, and used to train models for image classification.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Installation

Before running the code, install the necessary dependencies:

```sh
pip install kaggle py7zr tensorflow pandas matplotlib scikit-learn
```

## Dataset Setup

1. Configure Kaggle API:

```sh
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

2. Download CIFAR-10 dataset:

```sh
kaggle competitions download -c cifar-10
```

3. Extract dataset:

```python
from zipfile import ZipFile

dataset = 'cifar-10.zip'
with ZipFile(dataset, 'r') as zip:
    zip.extractall()
```

4. Extract training images:

```python
import py7zr

archive = py7zr.SevenZipFile('train.7z', mode='r')
archive.extractall()
archive.close()
```

## Data Preprocessing

- Load and process image files.
- Convert images to numpy arrays.
- Assign labels using a dictionary mapping class names to numbers.
- Perform train-test split (80-20 ratio).
- Normalize image pixel values by scaling them to [0,1].

## Model 1: Simple Neural Network

A basic feedforward neural network is implemented with the following structure:

- Input layer: Flattened 32x32x3 images
- Hidden layer: Dense (64 neurons, ReLU activation)
- Output layer: Dense (10 neurons, Softmax activation)

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32,32,3)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
model.fit(X_train_scaled, Y_train, validation_split=0.1, epochs=10)
```

## Model 2: ResNet50

ResNet50 is fine-tuned for CIFAR-10 image classification:

- Uses an upsampling layer to adjust input size.
- Incorporates batch normalization and dropout for regularization.

```python
from tensorflow.keras.applications.resnet50 import ResNet50

convolutional_base = ResNet50(weights='imagenet', include_top=False, input_shape=(256,256,3))
model = models.Sequential([
    layers.UpSampling2D((2,2)), layers.UpSampling2D((2,2)), layers.UpSampling2D((2,2)),
    convolutional_base, layers.Flatten(), layers.BatchNormalization(),
    layers.Dense(128, activation='relu'), layers.Dropout(0.5),
    layers.BatchNormalization(), layers.Dense(64, activation='relu'), layers.Dropout(0.5),
    layers.BatchNormalization(), layers.Dense(10, activation='softmax')
])
model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='sparse_categorical_crossentropy', metrics=['acc'])
```

## Model Evaluation

- The model is trained for 10 epochs.
- Accuracy and loss values are plotted.

```python
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

plt.plot(history.history['acc'], label='train accuracy')
plt.plot(history.history['val_acc'], label='validation accuracy')
plt.legend()
plt.show()
```

## Results

- The test accuracy is evaluated using:

```python
loss, accuracy = model.evaluate(X_test_scaled, Y_test)
print('Test Accuracy =', accuracy)
```

| Model                 | Test Accuracy                    |
| --------------------- | -------------------------------- |
| Custom Neural Network | 40.80% (Low due to fewer epochs) |
| ResNet50              | 93.86%                           |
