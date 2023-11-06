import numpy as np
import glob
import cv2
 
img_array = []

filename = 'left-steer.png'
img = cv2.imread(filename)
height, width, layers = img.shape
size = (width,height)

out = cv2.VideoWriter('left-steer.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

for i in range(30):
    out.write(img)
out.release()    
