import cv2
import numpy as np


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27: # 按下Esc键拍照
        cv2.imwrite('photo.jpg', frame)
        break

cap.release()
cv2.destroyAllWindows()
