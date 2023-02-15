# ROS2 imports 
import rclpy
from rclpy.node import Node

import numpy as np
import os, sys, cv2, time

# CV Bridge and message imports
from sensor_msgs.msg import Image as Image
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge, CvBridgeError

class Video2Img(Node):

    def __init__(self):
        super().__init__('videotoimage_node')
        
        self.declare_parameter("video_file", "/home/jetson/Data/Data/Video/GRMN0119.MP4")
        video_file = self.get_parameter("video_file").get_parameter_value().string_value
        # Create an Image publisher for the results
        self.publisher = self.create_publisher(Image,'camera/color/image_raw',10)

        # video_file = '/home/jetson/Data/Data/Video/GRMN0119.MP4'
        self.cap = cv2.VideoCapture(video_file)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        timer_period = 1/fps
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.bridge = CvBridge()
        
    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            resize_frame = cv2.resize(frame, (848,480), interpolation = cv2.INTER_AREA) #width 848, height 480
            self.publisher.publish(self.bridge.cv2_to_imgmsg(resize_frame, encoding="bgr8"))
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.get_logger().info('Publish video frame')

        

