import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import style
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, ZeroPadding2D, LeakyReLU, Flatten, Dropout,\
                                    BatchNormalization, Reshape, Conv2DTranspose

style.use("fivethirtyeight")

GENDER = ["Male", "Female"]
RACE = ["White", "Black", "Asian", "Indian", "Other"]


IMG_SIZE = 64
CHANELS = 3
TESTING = True

if TESTING:
    if CHANELS == 1:
        X = np.load("Data/FaceDataIMG_grey_small.npy")
    else:
        X = np.load("Data/FaceDataIMG_FFHQ.npy")
else:
    if CHANELS == 1:
        X = np.load("Data/FaceDataIMG_grey.npy")
    else:
        X = np.load("Data/FaceDataIMG_colour_big.npy")

print("Data loaded...")
X = (X-127.5)/127.5
print("Data normalized...")
np.random.shuffle(X)
print("Data shuffled...")
plt.ion()

FIG, AX = plt.subplots()
AX.grid(False)
IMG = None
if CHANELS == 1:
    CMAP = "gray"
    SHAPE = (IMG_SIZE, IMG_SIZE)
else:
    CMAP = None
    SHAPE = (IMG_SIZE, IMG_SIZE, 3)

LR = 1e-4
EPOCHS = 5
LABEL_SOFTENING = 0.2
SEED_SIZE = 100
DROPOUT = 0.25
BUFFER_SIZE = X.shape[0]
BATCH_SIZE = 64
PREV = True
train_dataset = tf.data.Dataset.from_tensor_slices(X).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
IMG_NOISE = np.random.randn(1, SEED_SIZE).astype("float32")

GEN_LOSS = []
DISC_LOSS = []

#  -------------------------------Discriminator Model----------------------------------------------

def make_discriminator_model_colour():
    model = keras.Sequential([Conv2D(128, (5,5), padding="same", input_shape=(IMG_SIZE, IMG_SIZE, CHANELS)),
                              LeakyReLU(),
                              MaxPool2D(2,2),
                              BatchNormalization(),
                              Dropout(DROPOUT),

                              Conv2D(128, (5, 5), padding="same"),
                              LeakyReLU(),
                              MaxPool2D(2, 2),
                              BatchNormalization(),
                              Dropout(DROPOUT),

                              Conv2D(128, (5, 5), padding="same"),
                              LeakyReLU(),
                              MaxPool2D(2, 2),
                              BatchNormalization(),
                              Dropout(DROPOUT),

                              Conv2D(128, (3, 3), padding="same"),
                              LeakyReLU(),
                              MaxPool2D(2, 2),
                              BatchNormalization(),
                              Dropout(DROPOUT),


                              Flatten(),

                              Dense(50, activation="relu"),
                              Dense(1)])
    return model

#discriminator = make_discriminator_model_colour()
discriminator = keras.models.load_model("Models/Disc_FFHQ.h5")
discriminator_optimizer = tf.optimizers.Nadam(LR)

def soft_ones():
    labels = []
    for i in range(BATCH_SIZE):
        labels.append(random.uniform(1-LABEL_SOFTENING, 1))
    return tf.convert_to_tensor(labels)
def soft_zeros():
    labels = []
    for i in range(BATCH_SIZE):
        labels.append(random.uniform(0, LABEL_SOFTENING))
    return tf.convert_to_tensor(labels)


def get_discriminator_loss(real_predictions, fake_predictions):
    real_predictions = tf.sigmoid(real_predictions)
    fake_predictions = tf.sigmoid(fake_predictions)

    real_loss = tf.losses.binary_crossentropy(soft_zeros(), real_predictions)
    fake_loss = tf.losses.binary_crossentropy(soft_ones(), fake_predictions)
    return fake_loss + real_loss

#  ---------------------------Generator Model--------------------------------------

def make_generator_model_colour():
    model = keras.Sequential([Dense(8*8*256, input_shape=(SEED_SIZE,)),
                              LeakyReLU(),
                              BatchNormalization(),
                              Reshape((8,8,256)),

                              Conv2DTranspose(128, (3,3), padding="same"),
                              LeakyReLU(),
                              BatchNormalization(),

                              Conv2DTranspose(128, (3,3), strides=(2,2), padding="same"),
                              LeakyReLU(),
                              BatchNormalization(),

                              Conv2DTranspose(128, (3, 3), padding="same"),
                              LeakyReLU(),
                              BatchNormalization(),

                              Conv2DTranspose(128, (5, 5), strides=(2,2), padding="same"),
                              LeakyReLU(),
                              BatchNormalization(),

                              Conv2DTranspose(CHANELS, (5,5), strides=(2, 2), padding="same", activation="tanh"),
                              ])
    return model

#generator = make_generator_model_colour()
generator = keras.models.load_model("Models/Generator_FFHQ.h5")
generator_optimizer = tf.optimizers.Nadam(LR)

def get_generator_loss(fake_predictions):
    fake_predictions = tf.sigmoid(fake_predictions)
    fake_loss = tf.losses.binary_crossentropy(tf.zeros_like(fake_predictions), fake_predictions)
    return fake_loss


#  ---------------------------------------Training-----------------------------------------------

def train_step(img, step, epoch):
    fake_img_noise = np.random.randn(BATCH_SIZE, SEED_SIZE).astype("float32")
    try:
        img = tf.reshape(img, (BATCH_SIZE, IMG_SIZE, IMG_SIZE, CHANELS))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(fake_img_noise)
            real_output = discriminator(img)
            fake_output = discriminator(generated_images)

            gen_loss = get_generator_loss(fake_output)
            disc_loss = get_discriminator_loss(real_output, fake_output)

            GEN_LOSS.append(np.mean(gen_loss))
            DISC_LOSS.append(np.mean(disc_loss))

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            if step % 50 == 0:
                perc = round((step/(BUFFER_SIZE/BATCH_SIZE))*100, 2)
                print(f"Epoch: {epoch+1}  {perc}%")
                print(f"G: {round(np.mean(gen_loss), 5)}  D: {round(np.mean(disc_loss), 5)}")

    except:
        pass


def plot():
    global IMG
    global AX
    global FIG

    pic = generator(IMG_NOISE)
    to_show = np.reshape(pic, SHAPE)
    y = np.copy(to_show)
    y += 1
    y /= 2

    if IMG is None:
        IMG = AX.imshow(y, CMAP)
    else:
        IMG.set_data(y)
    plt.pause(0.01)


def train(dataset, epochs):
    for i in range(epochs):
        step = 0
        print(f"----------------Epoch {i+1}----------------")
        try:
            for img in dataset:
                img = tf.cast(img, tf.dtypes.float32)
                train_step(img, step, i)
                step += 1
                if PREV:
                    plot()
        except:
            print("ERROR!!!!")




train(train_dataset, EPOCHS)

if TESTING:
    generator.save("Models/Generator_FFHQ.h5")
    discriminator.save("Models/Disc_FFHQ.h5")
else:
    generator.save("Models/Generator_colour_big.h5")
print("Done")

X_AXIS = np.linspace(0, len(DISC_LOSS), len(DISC_LOSS))
a = plt.plot(X_AXIS, DISC_LOSS, label="Discriminator")
b = plt.plot(X_AXIS, GEN_LOSS, label="Generator")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Loss")
plt.title("GANs for Faces")
plt.show()