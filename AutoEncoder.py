import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Reshape, Conv2DTranspose, BatchNormalization

IMG_SIZE = 128
CODE_SIZE = 100
DATA = np.load("Pics.npy")
print("Data loaded...")


def get_encoder():
    model = keras.Sequential([Conv2D(128, (5,5), padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
                              MaxPool2D(2, 2),

                              Conv2D(128, (5, 5), padding="same"),
                              MaxPool2D(2, 2),

                              Conv2D(128, (5, 5), padding="same"),
                              MaxPool2D(2, 2),

                              Conv2D(128, (3, 3), padding="same"),
                              Flatten(),

                              Dense(CODE_SIZE, activation="relu")])
    return model


ENCODER = get_encoder()

print("Encoding...")
CODES = []
count = 0
for pic in DATA:
    pic = tf.reshape(pic, (1, IMG_SIZE, IMG_SIZE, 1))
    code = ENCODER.predict(pic)
    CODES.append(code)
    count += 1
    print(round((count/len(DATA))*100, 2), "%")
print("Encoded...")
np.save("Codes.npy", CODES)
