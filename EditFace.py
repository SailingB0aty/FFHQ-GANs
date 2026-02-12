import pygame
import matplotlib.pyplot as plt
import random
import math
import numpy as np
from tensorflow.keras import models

pygame.init()
pygame.font.init()

my_font = pygame.font.SysFont('Comic Sans MS', 20)
MODEL = models.load_model("Models/Generator_FFHQ.h5")
WINDOW_SIZE = (1000, 800)
window = pygame.display.set_mode(WINDOW_SIZE)
RUN = True
FPS = 60
WEIGHTS = np.random.randn(100).astype("float32")
print(WEIGHTS)
FIG, AX = plt.subplots()
AX.grid(False)
IMG = None

background = (75, 75, 75)
slider = (255, 255, 255)
knob = (255, 0, 0)
text = (255, 255, 255)
def draw():
    window.fill(background)
    count = 0
    for y in range(5):
        for x in range(20):
            text_surface = my_font.render(str(y*20 + x + 1), False, text)
            pygame.draw.rect(window, slider, (x*40 + 50, y*150 + 50, 5, 100))
            pygame.draw.rect(window, knob, (x*40 + 48, (y*150 + 96 + WEIGHTS[count]*20), 9, 8))
            window.blit(text_surface, (x*40 + 40, y*150 + 20))
            count += 1

    pygame.display.update()


mouse_down = False
grabbed = None
prev_mouse = None
def get_input():
    global mouse_down
    global grabbed
    global prev_mouse
    mouse = pygame.mouse
    pressed = pygame.key.get_pressed()
    pos = mouse.get_pos()

    if mouse.get_pressed()[0] == 1:
        if not mouse_down:
            mouse_down = True
            grabbed_vector = (math.floor((pos[0]-25)/40), math.floor((pos[1]-25)/150))
            grabbed = grabbed_vector[1]*20 + grabbed_vector[0]
            prev_mouse = pos

        else:
            change = (prev_mouse[1] - pos[1])/20
            if WEIGHTS[grabbed] - change < 2.5 and WEIGHTS[grabbed] - change > -2.5:
                WEIGHTS[grabbed] -= change
            prev_mouse = pos
    else:
        mouse_down = False
        grabbed = None

    if pressed[pygame.K_UP]:
        for weight in WEIGHTS:
            if weight <= 2.49:
                weight += 0.1
    elif pressed[pygame.K_DOWN]:
        for weight in WEIGHTS:
            if weight >= -2.49:
                weight -= 0.1




def graph():
    global IMG
    global AX
    global FIG

    input = np.reshape(WEIGHTS, (1, 100))
    pic = MODEL(input)
    to_show = np.reshape(pic, (64, 64, 3))
    y = np.copy(to_show)
    y += 1
    y /= 2

    if IMG is None:
        IMG = AX.imshow(y)
    else:
        IMG.set_data(y)
    plt.pause(0.05)

clock = pygame.time.Clock()
while RUN:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            RUN = False

    get_input()
    draw()
    graph()


pygame.quit()
