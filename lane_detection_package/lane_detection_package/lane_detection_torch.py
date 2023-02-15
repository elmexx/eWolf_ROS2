#!/usr/bin/env python3

from __future__ import division

# Python imports
import numpy as np
import os, sys, cv2, time
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import pickle as pkl

import socket
import sys

# ROS imports
import rospy
import std_msgs.msg
from rospkg import RosPack
from sensor_msgs.msg import CompressedImage

# Lane Detection imports
from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
import lanecluster

package = RosPack()
package_path = package.get_path('lane_detection_node')

def predict_img(net,full_img,device,scale_factor=1,out_threshold=0.5):

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img)
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)            
        # probs = torch.sigmoid(output)
        probs = probs.squeeze(0)        
        tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ])
        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold

# Detector manager
class LaneDetector():
    def __init__(self):
        # Load weights parameter
        model_name = rospy.get_param('~model_name', 'pkl_1.4model.pth')
        self.model_path = os.path.join(package_path, 'model', model_name)
        rospy.loginfo("Found Model, loading %s", self.model_path)

        # Raise error if it cannot find the model
        if not os.path.isfile(self.model_path):
            raise IOError(('{:s} not found.').format(self.model_path))

        # Load parameter
        self.image_topic = rospy.get_param('~image_topic', '/usb_cam/image_raw')
        self.publish_image_topic = rospy.get_param('~publish_image_topic', '/lanebinary/image/compressed')
        #self.host = rospy.get_param('~speedgoat_ip','169.254.18.189')
        #self.port = ros.get_param('~speedgoat_port', 5500)

        # Load other parameters
        self.publish_image = rospy.get_param('~publish_image')
        
        # Load net
        self.model = UNet(n_channels=3, n_classes=1)
        with open(self.model_path, 'rb') as f:
            info_dict = pkl.load(f)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # rospy.loginfo(f'Using device {device}')
        self.model.to(device=self.device)
        self.model.load_state_dict(info_dict)
        # self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval() # Set in evaluation mode
        rospy.loginfo("Deep neural network loaded")
                
        # Define subscribers
        self.image_sub = rospy.Subscriber(self.image_topic, CompressedImage, self.imageCb, queue_size = 1)

        # Define publishers
        self.pub_viz_ = rospy.Publisher(self.publish_image_topic, CompressedImage, queue_size=10)
        rospy.loginfo("Launched node for object detection")

    def imageCb(self, ros_data):
        # Convert the image to OpenCV
        try:
            np_arr = np.frombuffer(ros_data.data, np.uint8)
            self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR).astype(np.uint8)
        except:
            rospy.loginfo("OpenCV don't get Image from ROS")
            self.cv_image = np.ones((256,512)).astype(np.uint8)
            
        self.img_pil = Image.fromarray(self.cv_image)
        mask = predict_img(net=self.model,full_img=self.img_pil,scale_factor=1,
                            out_threshold=0.2,device=self.device)
        binaryimg = (mask*255).astype(np.uint8)
#        print(binaryimg)
        self.binaryimg = cv2.resize(binaryimg, (512, 256))
        binaryimg_128 = cv2.resize(self.binaryimg,(256, 128)).astype(np.uint8)
        #lanemask, lane_coords = lanecluster.lane_mask_coords(self.binaryimg)
        #lanemaskresize = cv2.resize(lanemask,(self.cv_image.shape[1],self.cv_image.shape[0]))
        
#        cv2.namedWindow('binaryimg',cv2.WINDOW_AUTOSIZE)
#        cv2.imshow('binaryimg', cv2.resize(self.binaryimg,(512,256)))
        # Visualize detection results
        if (self.publish_image):
            self.visualizeAndPublish(self.binaryimg)

        return True

    def visualizeAndPublish(self,image):

        # Publish visualization image
        # image_msg = self.bridge.cv2_to_imgmsg(imgOut, "rgb8")
        cv2.namedWindow('binaryimg',cv2.WINDOW_AUTOSIZE)
        cv2.imshow('binaryimg', cv2.resize(image,(512,256)))
        cv2.waitKey(1)
        image_msg = CompressedImage()
        image_msg.header.stamp = rospy.Time.now()
        image_msg.format = "jpeg"
        image_msg.data = np.array(cv2.imencode('.jpg', image)[1]).tostring()
        self.pub_viz_.publish(image_msg)
       

if __name__=="__main__":
    # Initialize node
    rospy.init_node("lane_detector_node")

    # Define detector object
    dm = LaneDetector()

    # Spin
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Lane Detection module")
    cv2.destroyAllWindows()
    
