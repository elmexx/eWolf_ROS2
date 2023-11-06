# ROS2 imports 
import rclpy
from rclpy.node import Node

import numpy as np
import os, sys, cv2, time

# CV Bridge and message imports
from sensor_msgs.msg import Image as SensorImage
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge, CvBridgeError
import time


class ToImg(Node):

    def __init__(self):
        super().__init__('bagtoimage_node')

        self.declare_parameter("camera_topic", "/camera/color/image_raw")
        # Create a subscriber to the Image topic
        camera_topic = self.get_parameter("camera_topic").get_parameter_value().string_value
        self.get_logger().info("Subscription from %s" % camera_topic)
        
        self.subscription = self.create_subscription(SensorImage, camera_topic, self.listener_callback, 10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.idx = 1

    def listener_callback(self, data):
        # self.get_logger().info("Received an image! ")
        try:
          cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          cv_image = np.zeros((480,848)).astype(np.uint8)
          print(e)
        timestr = time.strftime("%Y%m%d_%H%M%S")
        # cv2.imwrite('/home/fmon005/Pictures/output/%06d.png'%self.idx,cv_image)
        cv2.imwrite('/home/fmon005/Pictures/output/%s.png'%timestr,cv_image)
        # self.idx = self.idx + 1
        

