import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

DIR = r"D:\Pictures"

IMG_SIZE = 64
COLOUR = True

X = []

count = 0

for folder in os.listdir(DIR):
    folder_dir = os.path.join(DIR, folder)
    try:
        for img in os.listdir(folder_dir):
            img_dir = os.path.join(folder_dir, img)
            try:
                if COLOUR:
                    to_add = cv2.imread(img_dir)
                    if to_add is not None:
                        if to_add.shape[0]/to_add.shape[1] <= 16/7 and to_add.shape[0]/to_add.shape[1] >= 16/11:
                            to_add = cv2.resize(to_add, (128, 72))
                            plt.imshow(cv2.cvtColor(to_add, cv2.COLOR_BGR2RGB))
                            plt.show()
                            X.append(cv2.cvtColor(to_add, cv2.COLOR_BGR2RGB))
                            count += 1
                else:
                    X.append(cv2.resize(cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE)))
            except:
               print("ERROR")
        print(f"Loaded {folder} -- Total: {count}")
    except:
        print("")

print(f"Loaded {count} images")

np.save("data/HomePics.npy", X)
