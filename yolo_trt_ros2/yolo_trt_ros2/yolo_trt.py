# ROS2 imports 
import rclpy
from rclpy.node import Node

# python imports
import numpy as np
import cv2
import os, sys, time

# pycuda imports
import pycuda.autoinit  # This is needed for initializing CUDA driver
import pycuda.driver as cuda

# CV Bridge and message imports
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import ObjectHypothesis, ObjectHypothesisWithPose, BoundingBox2D, Detection2D, Detection2DArray
import cv2
from cv_bridge import CvBridge, CvBridgeError
from bboxes_msg.msg import BoundingBox, BoundingBoxes

# Yolo Detection imports
from .utils.yolo_classes import get_cls_dict
from .utils.camera import add_camera_args, Camera
from .utils.display import open_window, set_display, show_fps
from .utils.visualization import BBoxVisualization
from .utils.yolo_with_plugins import TrtYOLO

WINDOW_NAME = 'TRT_Object_Detection'
VERBOSE=False

    
class YoloDetector(Node):

    def __init__(self):
        super().__init__('yolodetection_node')
        
        self.declare_parameter("camera_topic", "/camera/color/image_raw")
        # Create a subscriber to the Image topic
        camera_topic = self.get_parameter("camera_topic").get_parameter_value().string_value
        self.get_logger().info("Subscription from %s" % camera_topic)
        
        # Create a subscriber to the Image topic
        self.subscription = self.create_subscription(Image, camera_topic, self.listener_callback, 10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        # Create a Detection 2D array topic to publish results on
        self.detection_publisher = self.create_publisher(Detection2DArray, 'yolo2ddetection', 10)
        self.bboxes_publisher = self.create_publisher(BoundingBoxes, 'yolo2dbboxes', 10)
        # Create an Image publisher for the results
        self.result_publisher = self.create_publisher(Image,'yolodetection_image',10)
        
        model = 'yolov4-416'
        category_num = 80
        letter_box = False
        model_path = os.path.expanduser('~/ros2_ws/src/yolo_trt_ros2/yolo_trt_ros2/yolo/%s.trt' % model)

        if not os.path.isfile(model_path):
            raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % model)

        self.cls_dict = get_cls_dict(category_num)
        self.vis = BBoxVisualization(self.cls_dict)
        # h, w = get_input_shape(model)    
                
        # self.cuda_ctx = cuda.Device(0).make_context()
        self.trt_yolo = TrtYOLO(model, category_num, letter_box)            
        
        self.get_logger().info("YOLO Detection Model loaded /%s" % model)

    def listener_callback(self, data):

        try:
          cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          cv_image = np.ones((416,416)).astype(np.uint8)
          print(e)
        
        detection_array = Detection2DArray()
        bboxes_msg = BoundingBoxes()
        boxes, confs, clss = self.trt_yolo.detect(cv_image, conf_th=0.3)
                
        img = self.vis.draw_bboxes(cv_image, boxes, confs, clss)
        
        for box, cf, cl in zip(boxes, confs, clss):

            cl = int(cl)
            cls_name = self.cls_dict.get(cl, 'CLS{}'.format(cl))

            bbox_msg = BoundingBox()
            bbox_msg.probability = float(cf)
            bbox_msg.xmin = int(float(box[0]))
            bbox_msg.ymin = int(float(box[1]))
            bbox_msg.xmax = int(float(box[2]))
            bbox_msg.ymax = int(float(box[3]))
            bbox_msg.id = cl
            bbox_msg.cls = cls_name

            # Definition of 2D Bounding Boxes
            bboxes_msg.header = data.header
            bboxes_msg.bounding_boxes.append(bbox_msg)
            
            # Definition of 2D array message and ading all object stored in it.
            object_hypothesis_with_pose = ObjectHypothesisWithPose()
            object_hypothesis_with_pose.hypothesis.class_id = str(cls_name)
            object_hypothesis_with_pose.hypothesis.score = float(cf)

            bounding_box = BoundingBox2D()
            bounding_box.center.x = float((box[0] + box[2])/2)
            bounding_box.center.y = float((box[1] + box[3])/2)
            bounding_box.center.theta = 0.0
            
            bounding_box.size_x = float(2*(bounding_box.center.x - box[0]))
            bounding_box.size_y = float(2*(bounding_box.center.y - box[1]))

            detection = Detection2D()
            detection.header = data.header
            detection.results.append(object_hypothesis_with_pose)
            detection.bbox = bounding_box

            detection_array.header = data.header
            detection_array.detections.append(detection)  
               
        cv2.imshow(WINDOW_NAME, img)
        cv2.waitKey(1)
        
        # Publishing the results onto the the Detection2DArray vision_msgs format
        self.detection_publisher.publish(detection_array)
        self.bboxes_publisher.publish(bboxes_msg)
        # Publishing the results
        ros_image = self.bridge.cv2_to_imgmsg(img)
        ros_image.header.frame_id = 'yolo_frame'
        self.result_publisher.publish(ros_image)
        


