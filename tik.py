import numpy as np

import cv2

from utils.noise import random_blur, random_noise



img = cv2.imread('person/SeedLandImg/png_img/1572054596539.png')



while True:
  img_noised = random_blur(img)

  cv2.imshow('_', img_noised)
  key = cv2.waitKey(0)
  if key == 27:
    break


