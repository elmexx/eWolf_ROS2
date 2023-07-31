from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import Model, ClassAverages

# import TRT YOLO
import pycuda.autoinit  # This is needed for initializing CUDA driver
from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO

import os
import time

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg

import argparse

os.environ['TORCH_HOME'] = '/media/fmon005/Data/Data/Model'

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

parser.add_argument("--image-dir", default="eval/image_2/",
                    help="Relative path to the directory containing images to detect. Default \
                    is eval/image_2/")

# TODO: support multiple cal matrix input types
parser.add_argument("--cal-dir", default="eval/calib/",
                    help="Relative path to the directory containing camera calibration form KITTI. \
                    Default is camera_cal/")

parser.add_argument("--video", action="store_true",
                    help="Weather or not to advance frame-by-frame as fast as possible. \
                    By default, this will pull images from ./eval/video")

parser.add_argument("--show-yolo", action="store_true",
                    help="Show the 2D BoundingBox detecions on a separate image")

parser.add_argument("--hide-debug", action="store_true",
                    help="Supress the printing of each 3d location")


def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location

def main():

    # FLAGS = parser.parse_args()

    # load 3D detection model
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        print('Using previous model %s'%model_lst[-1])
        my_vgg = vgg.vgg19_bn(pretrained=True)
        model3D = Model.Model(features=my_vgg.features, bins=2).cuda()
        checkpoint = torch.load(weights_path + '/%s'%model_lst[-1])
        model3D.load_state_dict(checkpoint['model_state_dict'])
        model3D.eval()

    # load trt yolo v4
    model = 'yolov4-416'
    category_num = 80
    letter_box = False
    if not os.path.isfile(os.getenv("HOME")+ '/ros2_ws/src/yolo_trt_ros2/yolo_trt_ros2/yolo/%s.trt' % model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % model)
    cls_dict = get_cls_dict(category_num)
    # vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(model, category_num, letter_box) 
    
    averages = ClassAverages.ClassAverages()

    # TODO: clean up how this is done. flag?
    angle_bins = generate_bins(2)
    show_yolo = True

    img_path = '/home/fmon005/Downloads/color'#'./docs/image/'
    # imgfolder = '/home/fmon005/Downloads/color'
    # calib_path = '/home/fmon005/Documents/GKU/Stereo-RCNN-1.0/data/kitti/testing/calib'#'./docs/cal/'

    camera_info = np.array([[607.7498168945312, 0.0, 419.7865905761719],
                        [0.0, 606.7474365234375, 230.44131469726562],
                        [0.0, 0.0, 1.0]])
    proj_matrix = np.zeros((3, 4))
    proj_matrix[0:3, 0:3] = camera_info[0:3,0:3] #
    
    for filename in os.listdir(img_path):
        img_file = os.path.join(img_path, filename)
        # calib_file = os.path.join(calib_path, os.path.splitext(filename)[0]+'.txt')

        start_time = time.time()

        truth_img = cv2.imread(img_file)
        img = np.copy(truth_img)
        yolo_img = np.copy(truth_img)

        # detections = yolo.detect(yolo_img)
        # proj_matrix = get_calibration_cam_to_image(calib_file)
        
        boxes, confs, clss = trt_yolo.detect(yolo_img, conf_th=0.5)
        

        for box, cl in zip(boxes, clss):

            cl = int(cl)
            cls_name = cls_dict.get(cl, 'CLS{}'.format(cl))#
            print(cls_name)
            if not averages.recognized_class(cls_name):
                continue
            top_left = (box[0], box[1])
            bottom_right = (box[2], box[3])
            box_2d = [top_left, bottom_right]
            # this is throwing when the 2d bbox is invalid
            # TODO: better check
            try:
                detectedObject = DetectedObject(img, cls_name, box_2d, proj_matrix)
            except:
                print('DetectedObject error')
                continue

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            # box_2d = box
            detected_class = cls_name

            input_tensor = torch.zeros([1,3,224,224]).cuda()
            input_tensor[0,:,:,:] = input_img

            [orient, conf, dim] = model3D(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            if show_yolo:
                location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img)
            else:
                location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray)

            print('Estimated pose: %s' % location)
            print('time: ', time.time()-start_time)
            print('--------------')
        if show_yolo:
            numpy_vertical = np.concatenate((truth_img, img), axis=0)
            cv2.imshow('SPACE for next image, any other key to exit', numpy_vertical)
        else:
            cv2.imshow('3D detections', img)

        if cv2.waitKey(0) != 32: # space bar
            exit()

if __name__ == '__main__':
    main()
