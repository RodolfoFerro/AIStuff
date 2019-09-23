# ========================================================================
# Author: Rodolfo Ferro
# Contact: https://rodolfoferroxyz/
#
# Title: Intro to Machine Learning (ML Zero to Hero, part 2)
# From: https://www.youtube.com/watch?v=bemDFpNooA8
# ========================================================================


import tensorflow as tf
from tensorflow import keras
import numpy as np


# We load the fashion MNIST dataset:
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# We create a Keras sequential model:
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# We compile our model:
model.compile(optimizer=keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# We train the model:
model.fit(train_images, train_labels, epochs=5)

# We test the model with testing data:
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=False)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

# We can predict data from images:
prediction = model.predict(test_images[:1])
print(prediction)
