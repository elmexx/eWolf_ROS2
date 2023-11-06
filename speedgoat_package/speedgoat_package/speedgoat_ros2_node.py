# ROS2 imports 
import rclpy
from rclpy.node import Node

# python imports
import numpy as np
import socket
import cv2
import os, sys, time

# message imports
from sensor_msgs.msg import Image as SensorImage
from std_msgs.msg import String
from lane_parameter_msg.msg import LaneParams
from sensor_msgs.msg import CompressedImage
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer


def UDP_send(socket_UDP, server_address, msg):
    socket_UDP.sendto(msg,server_address)
    return None
    
class Speedgoat(Node):
    def __init__(self):
        super().__init__('speedgoat_node')
                
        left_sub = Subscriber(self, LaneParams, 'leftlanedetection')
        right_sub = Subscriber(self, LaneParams, 'rightlanedetection')
        # ts = TimeSynchronizer([left_sub, right_sub], 10)
        ts = ApproximateTimeSynchronizer([left_sub, right_sub], 10, 1)
        ts.registerCallback(self.callback)
        
        #self.host = "10.42.0.10"
        self.host = "169.254.18.189"
        self.port = 5500
        self.send_port = 8080
        self.buffersize = 1024
        self.server_address = (self.host, self.port) 
       
        # create Socket UDP
        self.socket_UDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket_UDP_receive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receive_addr = ("", self.send_port)
        self.socket_UDP_receive.bind(self.receive_addr)
        self.socket_UDP_receive.settimeout(1)
        
    def callback(self, left_sub, right_sub):
        # self.get_logger().info('left heard: "%f"' % left_sub.a)
        # self.get_logger().info('left heard: "%f"' % left_sub.b)
        # self.get_logger().info('right heard: "%f"' % right_sub.a)

        lane_params = np.array([left_sub.a,left_sub.b,left_sub.c, right_sub.a, right_sub.b, right_sub.c]).astype(np.float64)
        
        UDP_msg = lane_params.tobytes()
        # print(UDP_msg)
        UDP_send(socket_UDP=self.socket_UDP, server_address=self.server_address, msg=UDP_msg)
        # receive
        """
        try:
            recv_data, addr = self.socket_UDP_receive.recvfrom(64)
            outdata = np.frombuffer(recv_data,dtype=np.single)
            print('Receive Daten: ',outdata)
        except socket.timeout as e:
            pass   
        """     
        return True 

