import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

DIR = r"data/faces"
NP_FILENAME = "faces_64_RGB"

IMG_SIZE = 64
COLOUR = True

X = []

count = 0

# Data is to be contained within folders in the data\ directory
for folder in os.listdir(DIR):
    folder_dir = os.path.join(DIR, folder)
    try:
        for img in os.listdir(folder_dir):
            img_dir = os.path.join(folder_dir, img)
            try:
                if COLOUR:
                    to_add = cv2.imread(img_dir)
                    X.append(cv2.resize(cv2.imread(img_dir, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE)))
                    count += 1
                else:
                    X.append(cv2.resize(cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE)))
                    count += 1
            except:
               print("ERROR")
        print(f"Loaded {folder} -- Total: {count}")
    except:
        print("")

print(f"Loaded {count} images")

np.save(f"data/numpy/{NP_FILENAME}", X)
