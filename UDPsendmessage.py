# -*- coding: utf-8 -*-

import socket
import cv2
import numpy as np
import struct
 
#s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#
#img = np.ones((256,512)).astype(np.uint8)
#img_encode = cv2.imencode('.jpg', img)[1]
#data_encode = np.array(img_encode)
#data = data_encode.tobytes()
##定义文件头，打包成结构体
#fhead = struct.pack('l',len(data))
## 发送文件头:
#s.sendto(fhead,('127.0.0.1', 9999))
#counter = 0
##循环发送图片码流
#for i in range(len(data)//1024+1):
#    if 1024*(i+1)>len(data):
#        s.sendto(data[1024*i:], ('127.0.0.1', 9999))
#    else:
#        s.sendto(data[1024*i:1024*(i+1)], ('127.0.0.1', 9999))
#    counter = counter + 1
#    if counter > 10:
#        break
## 接收应答数据:
#print(s.recv(1024).decode('utf-8'))
##关闭
#s.close()

# Create a socket for sending data
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Convert the float to a binary representation
#value = np.array([3e-5,2e-1,3e2,1e-5,2e-1,-3e2]).astype(np.float64)
#data = value.tobytes()

# Send the binary data
while True:
    value = np.random.rand(1,6)
    value[:,5]=-value[:,5]
    data = value.tobytes()
    sock.sendto(data, ('localhost', 5500))

# Close the socket
sock.close()
