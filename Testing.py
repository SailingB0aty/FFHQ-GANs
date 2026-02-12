import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

COLOUR = False
SIZE = 64

if COLOUR:
    generator = keras.models.load_model("Models/Generator_FFHQ.h5")
    SHAPE = (SIZE, SIZE, 3)
else:
    generator = keras.models.load_model("Models/BestFaceModel.h5")
    SHAPE = (SIZE, SIZE)


while True:
    fake_img_noise = np.random.randn(1, 100).astype("float32")
    print(fake_img_noise)
    generated_image = generator(fake_img_noise)

    generated_image = np.reshape(generated_image, SHAPE)

    y = np.copy(generated_image)

    y += 1
    y /= 2

    plt.imshow(y, cmap="gray")
    plt.show()


print("Done")
