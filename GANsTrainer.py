import tensorflow as tf
import keras
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import style
from keras.layers import Conv2D, Dense, ReLU, MaxPool2D, ZeroPadding2D, LeakyReLU, Flatten, Dropout,\
                                    BatchNormalization, Reshape, Conv2DTranspose

style.use("fivethirtyeight")

GENDER = ["Male", "Female"]
RACE = ["White", "Black", "Asian", "Indian", "Other"]


IMG_SIZE = 64
CHANELS = 3
TESTING = False

# Define npy file names
RGB_NPY = "faces_64_RGB"
GRAY_NPY = ""

if TESTING:
    if CHANELS == 1:
        X = np.load("data/FaceDataIMG_grey_small.npy")
    else:
        X = np.load("data/FaceDataIMG_FFHQ.npy")
else:
    if CHANELS == 1:
        X = np.load(f"data/{GRAY_NPY}.npy")
    else:
        X = np.load(f"data/numpy/{RGB_NPY}.npy")

# Preprocess and shuffle data
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

GEN_LR = 2e-4
DISC_LR = 1e-4
EPOCHS = 50
LABEL_SOFTENING = 0.1
SEED_SIZE = 100
DROPOUT = 0.1
BUFFER_SIZE = X.shape[0]
BATCH_SIZE = 64
PREV = False
train_dataset = tf.data.Dataset.from_tensor_slices(X).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
IMG_NOISE = np.random.randn(1, SEED_SIZE).astype("float32")

GEN_LOSS = []
DISC_LOSS = []

bce = keras.losses.BinaryCrossentropy(from_logits=True)

#  -------------------------------Discriminator Model----------------------------------------------

def make_discriminator_model_colour():
    model = keras.Sequential([Conv2D(64, (5,5), padding="same", input_shape=(IMG_SIZE, IMG_SIZE, CHANELS)),
                              LeakyReLU(0.2),
                              Conv2D(filters=64, kernel_size=(4,4), strides=2, padding="same"),
                              Dropout(DROPOUT),

                              Conv2D(128, (5, 5), padding="same"),
                              LeakyReLU(0.2),
                              Conv2D(filters=128, kernel_size=(4,4), strides=2, padding="same"),
                              Dropout(DROPOUT),

                              Conv2D(256, (5, 5), padding="same"),
                              LeakyReLU(0.2),
                              Conv2D(filters=256, kernel_size=(4,4), strides=2, padding="same"),
                              Dropout(DROPOUT),

                              Conv2D(512, (3, 3), padding="same"),
                              LeakyReLU(0.2),
                              Conv2D(filters=512, kernel_size=(4,4), strides=2, padding="same"),
                              Dropout(DROPOUT),


                              Flatten(),
                              Dense(1),])
    return model

discriminator = make_discriminator_model_colour()
#discriminator = keras.models.load_model("Models/Disc_FFHQ.h5")
discriminator_optimizer = tf.optimizers.Adam(DISC_LR, beta_1=0.5, beta_2=0.999)

def soft_ones():
    labels = []
    for i in range(BATCH_SIZE):
        labels.append([random.uniform(1-LABEL_SOFTENING, 1)])
    return tf.convert_to_tensor(labels, dtype=tf.float32)
def soft_zeros():
    labels = []
    for i in range(BATCH_SIZE):
        labels.append([random.uniform(0, LABEL_SOFTENING)])
    return tf.convert_to_tensor(labels, dtype=tf.float32)


def get_discriminator_loss(real_predictions, fake_predictions):
    real_loss = bce(soft_ones(), real_predictions)
    fake_loss = bce(tf.zeros_like(fake_predictions), fake_predictions)
    return fake_loss + real_loss

#  ---------------------------Generator Model--------------------------------------

def make_generator_model_colour():
    model = keras.Sequential([Dense(4*4*256, input_shape=(SEED_SIZE,)),
                              BatchNormalization(),
                              ReLU(),
                              Reshape((4,4,256)),

                              Conv2DTranspose(128, (3,3), strides=(2,2), padding="same"),
                              BatchNormalization(),
                              ReLU(),

                              Conv2DTranspose(128, (3,3), strides=(2,2), padding="same"),
                              BatchNormalization(),
                              ReLU(),

                              Conv2DTranspose(128, (3, 3), padding="same"),
                              BatchNormalization(),
                              ReLU(),

                              Conv2DTranspose(128, (5, 5), strides=(2,2), padding="same"),
                              BatchNormalization(),
                              ReLU(),

                              Conv2DTranspose(CHANELS, (5,5), strides=(2, 2), padding="same", activation="tanh"),
                              ])
    return model

generator = make_generator_model_colour()
#generator = keras.models.load_model("Models/Generator_FFHQ.h5")
generator_optimizer = tf.optimizers.Adam(GEN_LR, beta_1=0.5, beta_2=0.999)

def get_generator_loss(fake_predictions):
    fake_loss = bce(tf.ones_like(fake_predictions), fake_predictions)
    return fake_loss


#  ---------------------------------------Training-----------------------------------------------

def train_step(img, step, epoch):
    fake_img_noise = np.random.randn(BATCH_SIZE, SEED_SIZE).astype("float32")
    try:
        img = tf.reshape(img, (BATCH_SIZE, IMG_SIZE, IMG_SIZE, CHANELS))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(fake_img_noise, training=True)
            real_output = discriminator(img, training=True)
            fake_output = discriminator(generated_images, training=True)

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

    except Exception as e:
        print("train_step error: ", repr(e))
        raise


def plot():
    global IMG
    global AX
    global FIG

    pic = generator(IMG_NOISE, training=False)
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
    generator.save("models/Generator_RGB.h5")
    discriminator.save("models/Discriminator_RGB.h5")
print("Done")

X_AXIS = np.linspace(0, len(DISC_LOSS), len(DISC_LOSS))
a = plt.plot(X_AXIS, DISC_LOSS, label="Discriminator")
b = plt.plot(X_AXIS, GEN_LOSS, label="Generator")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Loss")
plt.title("GANs for Faces")
plt.show()