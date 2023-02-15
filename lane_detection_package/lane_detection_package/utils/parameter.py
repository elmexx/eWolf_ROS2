import numpy as np

mtx = np.array([[1.121200e+03, 0.000000e+00, 9.634416e+02],
       [0.000000e+00, 1.131900e+03, 5.268709e+02],
       [0.000000e+00, 0.000000e+00, 1.000000e+00]]
)

dist = np.array([-0.3051, 0.0771, 0, 0, 0])

pitch = -1
yaw = -3
roll = 0
height = 1

distAheadOfSensor = 20
spaceToLeftSide = 4    
spaceToRightSide = 4
bottomOffset = 3

imgw = 1920
imgh = 1080