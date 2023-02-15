# ROS2 imports 
import rclpy
from rclpy.node import Node

import numpy as np
import os, sys, cv2, time
import pickle as pkl
import socket
from pykalman import KalmanFilter

# CV Bridge and message imports
from sensor_msgs.msg import Image as SensorImage
from std_msgs.msg import String, Header
from lane_parameter_msg.msg import LaneParams
#from builtin_interfaces.msg import Time
import cv2
from cv_bridge import CvBridge, CvBridgeError

# Torch import 
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import onnx
import onnxruntime as ort

# Lane Detection imports
from .unet import UNet
from .utils.data_vis import plot_img_and_mask
from .utils.dataset import BasicDataset

from .utils.lane_parameter import DictObjHolder, bev_perspective, lanefit, drawlane
#from .utils.lanecluster import lane_mask_coords

# garmin 1920*1080
"""
mtx = np.array([[1.121200e+03, 0.000000e+00, 9.634416e+02],
       [0.000000e+00, 1.131900e+03, 5.268709e+02],
       [0.000000e+00, 0.000000e+00, 1.000000e+00]])

dist = np.array([-0.3051, 0.0771, 0, 0, 0])
"""
# Intel realsense 848*480
mtx = np.array([[633.0128, 0., 425.0031],
                [0., 635.3088, 228.2753],
                [0.,0.,1.]
                ])
dist = np.array([0.1020, -0.1315, 0, 0, 0])

pitch = -2
yaw = 0
roll = 0
height = 1

distAheadOfSensor = 20
spaceToLeftSide = 4    
spaceToRightSide = 4
bottomOffset = 1

imgw = 848
imgh = 480


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
    
def predict_img_onnx(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.6):
    # net.eval()
    resizeimg = cv2.resize(full_img,(512,256))
            
    inputimg = np.expand_dims(np.rollaxis(resizeimg,axis=2,start=0),axis=0)

    output = net.run(None,
    {"inputImage": inputimg.astype(np.float32)},)
    output = np.array(output)[:,:,1,:,:]
        
    probs = torch.sigmoid(torch.Tensor(output))            
        # probs = torch.sigmoid(output)
    probs = probs.squeeze(0)

    tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(full_img.shape[0]),
            transforms.ToTensor()
        ]
    )

    probs = tf(probs.cpu())
    full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold
    
class LaneDetector(Node):

    def __init__(self):
        super().__init__('lanedetection_node')

        self.declare_parameter("camera_topic", "/camera/color/image_raw")
        # Create a subscriber to the Image topic
        camera_topic = self.get_parameter("camera_topic").get_parameter_value().string_value
        self.get_logger().info("Subscription from %s" % camera_topic)
        
        self.subscription = self.create_subscription(SensorImage, camera_topic, self.listener_callback, 10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

        # Create a Lane detection topic to publish results on
        self.leftdetection_publisher = self.create_publisher(LaneParams, 'leftlanedetection', 10)
        self.rightdetection_publisher = self.create_publisher(LaneParams, 'rightlanedetection', 10)

        # Create an Image publisher for the results
        self.result_publisher = self.create_publisher(SensorImage,'lanedetection_image',10)

        self.model = UNet(n_channels=3, n_classes=1)
        
        # Weights and labels locations
        
        self.model_path = os.getenv("HOME")+ '/ros2_ws/src/lane_detection_package/lane_detection_package/model/Lanenet.pth'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device=self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        self.get_logger().info("Lane Detection neural network loaded! ")
        """
        self.model_path = os.getenv("HOME")+ '/ros2_ws/src/lane_detection_package/lane_detection_package/model/lanenet.onnx'
        self.ort_session = ort.InferenceSession(self.model_path)
        self.get_logger().info("Lane Detection neural network loaded! ")
        """
        # Camera Pose
        self.CameraPose = DictObjHolder({
                "Pitch": pitch,
                "Yaw": yaw,
                "Roll": roll,
                "Height": height,
                })

        # IntrinsicMatrix
        self.IntrinsicMatrix = np.transpose(mtx)

        # Out Image View
        self.OutImageView = DictObjHolder({
                "distAheadOfSensor": distAheadOfSensor,
                "spaceToLeftSide": spaceToLeftSide,
                "spaceToRightSide": spaceToRightSide,
                "bottomOffset": bottomOffset,
                })

        self.OutImageSize = np.array([np.nan, np.int_(imgw/2)])  # image H, image W
        
        ###### Lane Kalmanfilter
        
        L_init_lane_parameter = np.array([-4.10852052e-05, -7.97958759e-02,  1.29887915e+02])
        R_init_lane_parameter = np.array([-2.79490979e-05,  4.62164141e-02,  2.44594161e+02])


        self.L_KF_init_parameter = np.array([L_init_lane_parameter[0],0,
                                        L_init_lane_parameter[1],0,
                                        L_init_lane_parameter[2],0])
        self.R_KF_init_parameter = np.array([R_init_lane_parameter[0],0,
                                        R_init_lane_parameter[1],0,
                                        R_init_lane_parameter[2],0])
        transition_matrix = np.eye(6)
        transition_matrix[0,1] = 1
        transition_matrix[2,3] = 1
        transition_matrix[4,5] = 1
        
        observation_matrix = np.zeros((3,6))
        observation_matrix[0,0] = 1
        observation_matrix[1,2] = 1
        observation_matrix[2,4] = 1
        
        state_covariance = 1*np.eye(6)
        observation_covariance = 50*np.eye(3)
        
        self.C_stateleft = self.L_KF_init_parameter
        self.C_stateright = self.R_KF_init_parameter
        
        self.P_stateleft = state_covariance
        self.P_stateright = state_covariance
        
        self.initial_state_mean = [0,0,0,0,0,0]
        self.initial_state_cov = np.eye(6)
        """
        self.L_kflane = KalmanFilter(transition_matrices = transition_matrix,
                              observation_matrices = observation_matrix,
                              transition_covariance  = state_covariance,
                              initial_state_mean = self.L_KF_init_parameter,
                              initial_state_covariance = self.initial_state_cov,
                              observation_covariance = observation_covariance)
                              
        self.R_kflane = KalmanFilter(transition_matrices = transition_matrix,
                              observation_matrices = observation_matrix,
                              transition_covariance  = state_covariance,
                              initial_state_mean = self.R_KF_init_parameter,
                              initial_state_covariance = self.initial_state_cov,                              
                              observation_covariance = observation_covariance)
        """                            
        self.L_kflane = KalmanFilter(transition_matrices = transition_matrix,
                              observation_matrices = observation_matrix,
                              initial_state_mean = self.L_KF_init_parameter,
                              initial_state_covariance = self.initial_state_cov,
                              observation_covariance = observation_covariance)
                              
        self.R_kflane = KalmanFilter(transition_matrices = transition_matrix,
                              observation_matrices = observation_matrix,
                              initial_state_mean = self.R_KF_init_parameter,
                              initial_state_covariance = self.initial_state_cov,                              
                              observation_covariance = observation_covariance)
                                                                 
    def listener_callback(self, data):
        # self.get_logger().info("Received an image! ")
        try:
          cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          cv_image = np.zeros((480,848)).astype(np.uint8)
          print(e)
        
        img_pil = Image.fromarray(cv_image)
        mask = predict_img(net=self.model,full_img=img_pil,scale_factor=1,
                            out_threshold=0.2,device=self.device)
        """                    
        mask = predict_img_onnx(net=self.ort_session, full_img=cv_image, scale_factor=1,
                           out_threshold=0.6, device=self.device)                    
        """                    
                            
        binaryimg = (mask*255).astype(np.uint8)

        binaryimg_original = cv2.resize(binaryimg, (imgw,imgh))
            
        # Bird's Eye View Image
        birdseyeviewimage, unwarp_matrix, birdseyeview = bev_perspective(cv_image, mtx, self.CameraPose, self.OutImageView, self.OutImageSize)
        # warpimage, _, _ = bev_perspective(binaryimg_original.astype(np.uint8), mtx, self.CameraPose, self.OutImageView, self.OutImageSize)

        # lane parameter in vehicle coordination
        lane_fit_params = lanefit(binaryimg_original,mtx,self.CameraPose, self.OutImageView, self.OutImageSize)
        ego_right_lane_o = lane_fit_params['ego_right_lane']
        ego_left_lane_o = lane_fit_params['ego_left_lane']
        ego_right_lane = lane_fit_params['right_new']
        ego_left_lane = lane_fit_params['left_new']
        
        # Tracking Lane with Kalman Filter
        # observation_covariance = 25*np.eye(3)
        """
        (self.C_stateleft, self.P_stateleft) = self.L_kflane.filter_update(filtered_state_mean = self.C_stateleft,
                                                                          filtered_state_covariance = self.P_stateleft,
                                                                          observation_covariance = observation_covariance,
                                                                          observation = ego_left_lane)

        ego_left_lane = np.array([self.C_stateleft[0], self.C_stateleft[2], self.C_stateleft[4]])   
        
        (self.C_stateright, self.P_stateright) = self.R_kflane.filter_update(filtered_state_mean = self.C_stateright,
                                                      filtered_state_covariance = self.P_stateright,
                                                      observation_covariance = observation_covariance,
                                                      observation = ego_right_lane)

        ego_right_lane = np.array([self.C_stateright[0], self.C_stateright[2], self.C_stateright[4]])
        """      
              
                
        (self.C_stateleft, self.P_stateleft) = self.L_kflane.filter_update(filtered_state_mean = self.C_stateleft if self.C_stateright is not None else self.L_KF_init_parameter,
                                                                          filtered_state_covariance = self.P_stateleft if self.P_stateleft is not None else self.initial_state_cov,
                                                                          observation = ego_left_lane)

        ego_left_lane = np.array([self.C_stateleft[0], self.C_stateleft[2], self.C_stateleft[4]])   
        
        (self.C_stateright, self.P_stateright) = self.R_kflane.filter_update(filtered_state_mean = self.C_stateright if self.C_stateright is not None else self.R_KF_init_parameter,
                                                      filtered_state_covariance = self.P_stateright if self.P_stateright is not None else self.initial_state_cov,
                                                      observation = ego_right_lane)

        ego_right_lane = np.array([self.C_stateright[0], self.C_stateright[2], self.C_stateright[4]])                
        
        mask_img = drawlane(binaryimg_original, birdseyeviewimage, unwarp_matrix, ego_right_lane_o, ego_left_lane_o)
        g_b = np.zeros_like(mask_img).astype(np.uint8)
        add_img = cv2.merge((g_b, g_b, mask_img))
        out_image = cv2.addWeighted(cv2.resize(cv_image,(imgw, imgh)),1,add_img,1,0.0)
        
        left_detection = LaneParams()
        right_detection = LaneParams()
        
        msg_stamp = self.get_clock().now().to_msg()
        
        left_detection.header.stamp = msg_stamp
        left_detection.header.frame_id =='left_lane'
        left_detection.a = ego_left_lane[0]
        left_detection.b = ego_left_lane[1]
        left_detection.c = ego_left_lane[2]
        
        right_detection.header.stamp = msg_stamp
        right_detection.header.frame_id =='right_lane'
        right_detection.a = ego_right_lane[0]
        right_detection.b = ego_right_lane[1]
        right_detection.c = ego_right_lane[2]
        
        # cv2.imshow('binaryimg', warpimage)
        cv2.imshow('outimage', cv2.resize(out_image,(512,256)))
        
        cv2.waitKey(1)
        
        # Publishing the results onto the the lane detection vision_msgs format
        self.leftdetection_publisher.publish(left_detection)
        self.rightdetection_publisher.publish(right_detection)
        
        ros_image = self.bridge.cv2_to_imgmsg(out_image)
        ros_image.header.frame_id = 'lane_detectimg'
        self.result_publisher.publish(ros_image)
        
        
        

