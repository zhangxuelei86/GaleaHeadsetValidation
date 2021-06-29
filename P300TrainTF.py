import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from utils import conv1d_output_shape

# Hyper-parameters
input_size = 35  # change this base on the length of x
hidden_size = 64
num_classes = 2
num_epochs = 100
batch_size = 128
learning_rate = 0.001
num_chan = 3

x = np.load('P300_062221-062521/data.npy')
x = x.reshape()
y = np.load('P300_062221-062521/labels.npy')
le = preprocessing.LabelEncoder()
le.fit(y)
y_encoded = le.transform(y)

# stratified split for balancing classes before and after train-test-split
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42, stratify=y)

model = models.Sequential()
model.add(layers.Conv1D(8, 5, strides=1, activation='relu', input_shape=(3, input_size), data_format='channels_last'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Flatten())
model.add(layers.Dense(hidden_size, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=num_epochs,
                    validation_data=(x_test, y_test))