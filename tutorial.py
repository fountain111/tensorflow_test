import tensorflow as tf


print(tf.__version__)
import cv2
import numpy

#import matplotlib.pyplot as plot

cap = cv2.VideoCapture(0)

while(1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break