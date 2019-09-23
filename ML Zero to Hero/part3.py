# ========================================================================
# Author: Rodolfo Ferro
# Contact: https://rodolfoferroxyz/
#
# Title: Intro to Machine Learning (ML Zero to Hero, part 3)
# From: https://www.youtube.com/watch?v=x_VrgWTKkiM
# ========================================================================


import tensorflow as tf
from tensorflow import keras
import numpy as np


# We load the fashion MNIST dataset:
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# We preprocess the input data to feed the neural network:
trn, trw, trh = train_images.shape
tsn, tsw, tsh = test_images.shape
train_images = train_images.reshape((trn, trw, trh, 1))
test_images = test_images.reshape((tsn, tsw, tsh, 1))

# We create a Keras sequential model:
model = keras.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# We compile our model:
model.compile(optimizer=keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# We train the model:
model.fit(train_images, train_labels, epochs=3)

# We test the model with testing data:
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=False)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

# We can predict data from images:
# prediction = model.predict(<your-image>)
# print(prediction)
