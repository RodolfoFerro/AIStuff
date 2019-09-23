# ========================================================================
# Author: Rodolfo Ferro
# Contact: https://rodolfoferroxyz/
#
# Title: Intro to Machine Learning (ML Zero to Hero, part 1)
# From: https://www.youtube.com/watch?v=KNAWp2S3w94
# ========================================================================


import tensorflow.keras as keras
import numpy as np


# We create a Keras sequential model:
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# We define the sample data from our function Y = 2X - 1:
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# We train the model:
model.fit(xs, ys, epochs=1000)

# We test the model with X = 10:
print(model.predict([10.0]))

