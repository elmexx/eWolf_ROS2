import numpy as np
import torch
import cv2
import os
import os.path as osp
import glob
import argparse
from lanedet.datasets.process import Process
from lanedet.models.registry import build_net
from lanedet.utils.config import Config
from lanedet.utils.visualization import imshow_lanes
from lanedet.utils.net_utils import load_network
from pathlib import Path
from tqdm import tqdm
from tools.lane_parameter import DictObjHolder, bev_perspective, PolynomialRegression, get_fit_param, insertLaneBoundary
from sklearn.linear_model import RANSACRegressor

import matplotlib.pyplot as plt

from numba import jit

# Intel realsense 848*480
mtx = np.array([[633.0128, 0., 425.0031],
                [0., 635.3088, 228.2753],
                [0.,0.,1.]
                ])
dist = np.array([0.1020, -0.1315, 0, 0, 0])

pitch = 3
yaw = 0
roll = 0
height = 1.6

distAheadOfSensor = 30
spaceToLeftSide = 4    
spaceToRightSide = 4
bottomOffset = 1

imgw = 848
imgh = 480

CameraPose = DictObjHolder({
        "Pitch": pitch,
        "Yaw": yaw,
        "Roll": roll,
        "Height": height,
        })

# IntrinsicMatrix
IntrinsicMatrix = np.transpose(mtx)

# Out Image View
OutImageView = DictObjHolder({
        "distAheadOfSensor": distAheadOfSensor,
        "spaceToLeftSide": spaceToLeftSide,
        "spaceToRightSide": spaceToRightSide,
        "bottomOffset": bottomOffset,
        })

OutImageSize = np.array([np.nan, np.int_(imgw/2)])  # image H, image W
left_fit_model = RANSACRegressor(PolynomialRegression(degree=2), residual_threshold=10)
right_fit_model = RANSACRegressor(PolynomialRegression(degree=2), residual_threshold=10)

class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.x_scale = self.cfg.sensor_img_w / self.cfg.ori_img_w
        self.y_scale = self.cfg.sensor_img_h / self.cfg.ori_img_h

        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(
                self.net, device_ids = range(1)).cuda()
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

    def preprocess(self, ori_img):
        # ori_img = cv2.imread(img_path)
        ori_img = cv2.resize(ori_img, (self.cfg.ori_img_w,self.cfg.ori_img_h))
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'ori_img':ori_img})
        return data

    def inference(self, data):
        with torch.no_grad():
            data = self.net(data)
            data = self.net.module.get_lanes(data)
        return data

    def show(self, data):
        out_file = self.cfg.savedir 
        if out_file:
            out_file = osp.join(out_file, 'img_path')
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        x_scale = self.cfg.sensor_img_w / self.cfg.ori_img_w
        y_scale = self.cfg.sensor_img_h / self.cfg.ori_img_h

        imshow_lanes(data['ori_img'], lanes, x_scale, y_scale, show=True, out_file=out_file)

    def run(self, data):
        data = self.preprocess(data)
        data['lanes'] = self.inference(data)[0]

        if self.cfg.show or self.cfg.savedir:
            self.show(data)
        return data
    

def get_img_paths(path):
    p = str(Path(path).absolute())  # os-agnostic absolute path
    if '*' in p:
        paths = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        paths = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        paths = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    return paths 

def process(args):
    cfg = Config.fromfile(args.config)
    cfg.show = False
    cfg.savedir = None
    cfg.load_from = args.load_from
    detect = Detect(cfg)
    paths = get_img_paths(args.img)
    for p in tqdm(paths):
        detect.run(p)

def moving_average(array, window_data):
    window_data[:-1] = window_data[1:]
    window_data[-1] = array
    moving_avg = np.mean(window_data, axis=0)
    return moving_avg

if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='The path of config file')
    parser.add_argument('--img',  help='The path of the img (img file or img_folder), for example: data/*.png')
    parser.add_argument('--show', action='store_true', 
            help='Whether to show the image')
    parser.add_argument('--savedir', type=str, default=None, help='The root of save directory')
    parser.add_argument('--load_from', type=str, default='best.pth', help='The path of model')
    args = parser.parse_args()
    process(args)
    """

    cfg = Config.fromfile("configs/condlane/resnet101_culane.py")
    cfg.show = False
    cfg.savedir = None
    cfg.load_from = "model/condlane_r101_culane.pth"
    detect = Detect(cfg)
    # paths = get_img_paths("Campus.png")
    # img = cv2.imread("Test.png")
    # ori_img = img.copy()
    # data = detect.run(img)
    # lanes = [lane.to_array(detect.cfg) for lane in data['lanes']]
    # lane_num = len(lanes)
    # warpimage, unwarp_matrix, birdseyeview = bev_perspective(img, mtx, CameraPose, OutImageView, OutImageSize)
    # lanes_list = [[] for _ in range(lane_num)]
    # lanes_wc = [[] for _ in range(lane_num)]
    # lane_idx = 0
    # for lane in lanes:
    #     for x, y in lane:
    #         if x <= 0 or y <= 0:
    #             continue
    #         x = np.round(x * detect.x_scale)
    #         y = np.round(y * detect.y_scale)
    #         x, y = int(x), int(y)
    #         cv2.circle(img, (x, y), 3, (0,255,0), -1)

    #         lanes_list[lane_idx].append(np.array([x,y]))
    #     lanes_wc[lane_idx], _ = birdseyeview.imagetovehicle(np.asarray(lanes_list[lane_idx]))
    #     lane_idx = lane_idx + 1
    # alpha = 0.4
    # img = cv2.addWeighted(img, alpha, ori_img, 1 - alpha, 0)
    # # get left and right lanes
    # left_lanes = []
    # right_lanes = []
    # if lanes_wc:    #   !!if lanes_wc = [[],[]], there will give an error!
    #     for lane in lanes_wc:
    #         lateraloffset = lane[0][1]
    #         if lateraloffset>=0:
    #             left_lanes.append(lane)
    #         else:
    #             right_lanes.append(lane)
    # if left_lanes:
    #     left_lateraloffset = []
    #     for lane in left_lanes: 
    #         left_lateraloffset.append(lane[0][1])
    #     idxmin = np.argmin(left_lateraloffset)
    #     egoleft = left_lanes[idxmin]
    # if right_lanes:
    #     right_lateraloffset = []
    #     for lane in right_lanes: 
    #         right_lateraloffset.append(lane[0][1])
    #     idxmax = np.argmax(right_lateraloffset)
    #     egoright = right_lanes[idxmax]
    # left_init_param = np.array([])
    # right_init_param = np.array([])
    # leftparam = get_fit_param(egoleft, left_init_param, left_fit_model)
    # rightparam = get_fit_param(egoright, right_init_param, right_fit_model)

    # left_lane_img = insertLaneBoundary(img, warpimage, leftparam, OutImageView, birdseyeview, (0,0,255))
    # lane_img = insertLaneBoundary(left_lane_img, warpimage, rightparam, OutImageView, birdseyeview, (255,0,0))
    window_size = 10
    left_window_data = np.zeros((window_size,3))
    right_window_data = np.zeros((window_size,3))
    video_path = 'test_campus_curve.mp4'
    cap = cv2.VideoCapture(video_path)

    m_point = 480

    while(cap.isOpened()):
        ret, img = cap.read()
        if ret==True:
            ori_img = img.copy()
            data = detect.run(img)
            lanes = [lane.to_array(detect.cfg) for lane in data['lanes']]
            lane_num = len(lanes)
            warpimage, unwarp_matrix, birdseyeview = bev_perspective(img, mtx, CameraPose, OutImageView, OutImageSize)
            lanes_list = [[] for _ in range(lane_num)]
            lanes_wc = [[] for _ in range(lane_num)]
            lane_idx = 0
            binaryimg_original = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
            for lane in lanes:
                for x, y in lane:
                    if x <= 0 or y <= 0:
                        continue
                    x = np.round(x * detect.x_scale)
                    y = np.round(y * detect.y_scale)
                    x, y = int(x), int(y)
                    cv2.circle(binaryimg_original, (x, y), 3, 255, -1)
                    # cv2.circle(img, (x, y), 3, (0,255,0), -1)

                    lanes_list[lane_idx].append(np.array([x,y]))
                lanes_wc[lane_idx], _ = birdseyeview.imagetovehicle(np.asarray(lanes_list[lane_idx]))
                lane_idx = lane_idx + 1
            alpha = 0.4
            img = cv2.addWeighted(img, alpha, ori_img, 1 - alpha, 0)
            
            ### 
            # if len(lanes_wc) == 1:
            #     lane = lanes_wc[0]
            #     lateraloffset = lane[0][1]
            #     if lateraloffset>=0:
            #         v_right = lane
            #         v_right[:,1] = v_right[:,1]-3.5
            #         lanes_wc.append(v_right)
            #     else:
            #         v_left = lane
            #         v_left[:,1] = v_left[:,1]+3.5
            #         lanes_wc.append(v_left)

            # get left and right lanes
            left_lanes = []
            right_lanes = []
            egoleft = np.array([])
            egoright = np.array([])
            if lanes_wc:    #   !!if lanes_wc = [[],[]], there will give an error!
                for lane in lanes_wc:
                    lateraloffset = lane[0][1]
                    if lateraloffset>=0:
                        left_lanes.append(lane)
                    else:
                        right_lanes.append(lane)
            if left_lanes:
                left_lateraloffset = []
                for lane in left_lanes: 
                    left_lateraloffset.append(lane[0][1])
                idxmin = np.argmin(left_lateraloffset)
                egoleft = left_lanes[idxmin]
                if egoleft[0][1] > 3:
                    egoleft = np.array([])
            if right_lanes:
                right_lateraloffset = []
                for lane in right_lanes: 
                    right_lateraloffset.append(lane[0][1])
                idxmax = np.argmax(right_lateraloffset)
                egoright = right_lanes[idxmax]
                if egoright[0][1] < -3:
                    egoright = np.array([])

            if egoleft.size==0 and egoright.size!=0:
                egoleft = egoright.copy()
                egoleft[:,1] = egoleft[:,1]+3.5
            if egoleft.size!=0 and egoright.size==0:
                egoright = egoleft.copy()
                egoright[:,1] = egoright[:,1]-3.5
            #####
            # get ego left and right lane
            #####
            # if left_lanes:
            #     left_lateraloffset = []
            #     for lane in left_lanes: 
            #         left_lateraloffset.append(lane[0][1])
            #     idxmin = np.argmin(left_lateraloffset)
            #     egoleft = left_lanes[idxmin]
            # if right_lanes:
            #     right_lateraloffset = []
            #     for lane in right_lanes: 
            #         right_lateraloffset.append(lane[0][1])
            #     idxmax = np.argmax(right_lateraloffset)
            #     egoright = right_lanes[idxmax]

            
            
            left_init_param = np.array([])
            right_init_param = np.array([])
            leftparam = get_fit_param(egoleft, left_init_param, left_fit_model)
            rightparam = get_fit_param(egoright, right_init_param, right_fit_model)

            if leftparam.size > 0:
                leftparam = moving_average(leftparam, left_window_data)
                left_window_data = np.vstack((left_window_data[1:], leftparam))
            if rightparam.size > 0:
                rightparam = moving_average(rightparam, right_window_data)
                right_window_data = np.vstack((right_window_data[1:], rightparam))

            m_laneparam = (leftparam + rightparam)/2
            left_lane_img = insertLaneBoundary(img, warpimage, leftparam, OutImageView, birdseyeview, (0,0,255))
            lane_img = insertLaneBoundary(left_lane_img, warpimage, rightparam, OutImageView, birdseyeview, (255,0,0))
            m_lane_img = insertLaneBoundary(lane_img, warpimage, m_laneparam, OutImageView, birdseyeview, (0,255,255), True)
            
            out_img = cv2.line(m_lane_img, (480,460),(480,480),(0,255,0),3)

            cv2.imshow('condlane', out_img)
            # cv2.imshow('bev', binaryimg_original)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

        
