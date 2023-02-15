#!/usr/bin/env python3
"""speedgoat_jetson_ehternet.py

This script demonstrates how to communicate between speedgoat and jetson with ethernet
"""

from __future__ import division

# Python imports
import numpy as np
import socket
import cv2


# ROS imports
import rospy
import std_msgs.msg
from rospkg import RosPack
from sensor_msgs.msg import CompressedImage

package = RosPack()
package_path = package.get_path('speedgoat_node')

def UDP_send(socket_UDP, server_address, msg):
    socket_UDP.sendto(msg,server_address)
    return None

# communicate between speedgoat and jetson
class Speedgoat():
    def __init__(self):

        # Load topic parameter
        self.listen_image_topic = rospy.get_param("/speedgoat_ros/listen_image_topic")
        
        self.host = rospy.get_param("/speedgoat_ros/host_ip")
        # "169.254.18.189"
        self.port = rospy.get_param("/speedgoat_ros/host_port")
        self.send_pot = rospy.get_param("/speedgoat_ros/send_port")
        self.buffersize = 1024
        self.server_address = (self.host, self.port) 
        
        # create Socket UDP
        self.socket_UDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket_UDP_receive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receive_addr = ("", self.send_pot)
        self.socket_UDP_receive.bind(self.receive_addr)
        self.socket_UDP_receive.settimeout(1)
        
        # Define subscribers
        self.image_sub = rospy.Subscriber(self.listen_image_topic, CompressedImage, self.send_recv_msg, queue_size = 1)

        # Define publishers
        # self.pub_ = rospy.Publisher(self.publish_image_topic, CompressedImage, queue_size=10)
        rospy.loginfo("Launched node for Speedgoat Node")

    def send_recv_msg(self, ros_data):
        # Convert the image to OpenCV
        try:
            np_arr = np.frombuffer(ros_data.data, np.uint8)
            self.lane_image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        except:
            rospy.loginfo("OpenCV don't get Image from ROS")
            self.lane_image = np.ones((128,256)).astype(np.uint8)

        binaryimg_128 = cv2.resize(self.lane_image,(256, 128)).astype(np.uint8)

        UDP_msg = binaryimg_128.tobytes()
        
        UDP_send(socket_UDP=self.socket_UDP, server_address=self.server_address, msg=UDP_msg)
        # receive
        try:
            recv_data, addr = self.socket_UDP_receive.recvfrom(64)
            outdata = np.frombuffer(recv_data,dtype=np.single)
            print('Receive Daten: ',outdata)
        except socket.timeout as e:
            pass            

        # image_msg = CompressedImage()
        # image_msg.header.stamp = rospy.Time.now()
        # image_msg.format = "jpeg"
        # image_msg.data = np.array(cv2.imencode('.jpg', binaryimg_128)[1]).tostring()
        # self.pub_viz_.publish(image_msg)
        return True
       

if __name__=="__main__":
    # Initialize node
    rospy.init_node("speedgoat_ros")

    # Define object
    sg = Speedgoat()

    # Spin
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Speedgoat module")
    
