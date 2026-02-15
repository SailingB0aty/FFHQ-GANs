# GAN Faces (2019) #
This legacy passion project uses the Generative Adversarial Networks method to train a model that can generate faces.

The generator model takes an array of 100 floats (-1 < n < 1) and generates a 64x64 RGB image.

## Demo ##
https://github.com/user-attachments/assets/817d532a-2ce8-456f-b523-9f296b43f9b3

## Data ##
The dataset used to train is NVlabs ffhq-dataset scaled down to 64*64 pixels. Find the dataset at: https://github.com/NVlabs/ffhq-dataset

## Linux setup ##
```bash
python3 venv venv
source venv/bin/activate
pip install -r requirements.txt
```
