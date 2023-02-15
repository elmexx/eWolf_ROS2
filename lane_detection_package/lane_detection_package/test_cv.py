
import numpy as np
import os, sys, cv2, time

video_file = '/home/jetson/Data/Data/Video/GRMN0119.MP4'
cap = cv2.VideoCapture(video_file)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()

