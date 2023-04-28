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

class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg
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
        # x_scale = self.cfg.sensor_img_w / self.cfg.ori_img_w
        # y_scale = self.cfg.sensor_img_h / self.cfg.ori_img_h
        # print(len(lanes))
        # imshow_lanes(data['ori_img'], lanes, x_scale, y_scale, show=self.cfg.show, out_file=out_file)
        imshow_lanes(data['ori_img'], lanes, self.x_scale, self.y_scale, show=True, out_file=out_file)

    def run(self, data):
        data = self.preprocess(data)
        data['lanes'] = self.inference(data)[0]
        self.x_scale = self.cfg.sensor_img_w / self.cfg.ori_img_w
        self.y_scale = self.cfg.sensor_img_h / self.cfg.ori_img_h
        if self.cfg.show or self.cfg.savedir:
            self.show(data, self.x_scale, self.y_scale)

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
    # cfg.show = args.show
    cfg.show = False
    cfg.savedir = args.savedir
    cfg.load_from = args.load_from
    detect = Detect(cfg)
    # paths = get_img_paths(args.video)
    video_path = args.video
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, ori_img = cap.read()
        if ret==True:
            data = detect.run(ori_img)
            detect.show(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='The path of config file')
    parser.add_argument('--video',  help='The path of the video (img file or img_folder), for example: data/*.png')
    parser.add_argument('--show', action='store_true', 
            help='Whether to show the image')
    parser.add_argument('--savedir', type=str, default=None, help='The root of save directory')
    parser.add_argument('--load_from', type=str, default='best.pth', help='The path of model')
    args = parser.parse_args()
    process(args)
