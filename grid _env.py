import cv2
import numpy as np
import matplotlib.pyplot as plt
# import tkinter as tk
import time
import random as rd
import sys

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27: # 按下Esc键拍照
        cv2.imwrite('photo.jpg', frame)
        break

cap.release()
cv2.destroyAllWindows()

UNIT = 20  # 像素值

img = cv2.imread('photo.jpg', cv2.IMREAD_GRAYSCALE)

threshold_value = 128
max_value = 255
ret, threshold_image = cv2.threshold(img, threshold_value, max_value, cv2.THRESH_BINARY)

cv2.imshow('Threshold Image', threshold_image)

cv2.waitKey(0)
cv2.destroyAllWindows()