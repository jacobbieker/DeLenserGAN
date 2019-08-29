import h5py
import numpy as np
from redshift_loader import DataLoader
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
import keras

from keras.optimizers import Adam

# Input shape
img_rows = 64
img_cols = 64
channels = 5
img_shape = (img_rows, img_cols, channels)

# Configure data loader
data = np.loadtxt("network/networkFrame.csv", delimiter=',', dtype=str)

train = DataLoader(img_res=(img_rows, img_cols), batch_size=32, norm=False, data=data[:int(len(data)/2)])
val = DataLoader(img_res=(img_rows, img_cols), batch_size=32, norm=False, data=data[int(len(data)/2):])

model = Sequential()
model.add(Conv2D(32, input_shape=img_shape, kernel_size=3, strides=2, padding='same'))
model.add(Activation(LeakyReLU()))
model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
model.add(Activation(LeakyReLU()))
model.add(MaxPooling2D())
model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
model.add(Activation(LeakyReLU()))
model.add(MaxPooling2D())
model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
model.add(Activation(LeakyReLU()))
#model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(1, activation='linear'))

model.compile(loss="mse", optimizer="adam", metrics=["mae"])

model.summary()
callbacks = list()
callbacks.append(keras.callbacks.TensorBoard(log_dir="./", histogram_freq=0, write_graph=True))
callbacks.append(
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-10))
callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=20))
callbacks.append(keras.callbacks.TerminateOnNaN())

model.fit_generator(
    generator=train,
    epochs=500,
    verbose=2,
    validation_data=val,
    callbacks=callbacks,
    use_multiprocessing=False,
    workers=1,
    max_queue_size=100,
)
