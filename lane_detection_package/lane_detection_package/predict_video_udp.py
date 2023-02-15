# -*- coding: utf-8 -*-

import argparse
import logging
import os
import cv2

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pykalman import KalmanFilter
import matplotlib.pyplot as plt

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset

import parameter
import image2bev
import lane_parameter
import lanecluster

from torch2trt import torch2trt
from torch2trt import TRTModule

import socket
import sys

DT = 0.2
HOST = '169.254.18.189'  
PORT = 5500
buffersize = 1024
server_address = (HOST, PORT) 

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    # net.eval()

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
        
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', 
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def get_waypoints(lane_params):
    way_points = lane_fit_params['waypoints']
    way_points = np.fliplr(way_points)
    return way_points


# client
def UDP_send(socket_UDP, server_address, msg):
    # socket_UDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    socket_UDP.sendto(msg,server_address)
    # data, addr = socket_UDP.recvfrom(buffersize)
    # outdata = np.frombuffer(data,dtype=np.double)
    return None

def UDP_receive(socket_UDP):
    buffersize = 1024
    data, addr = socket_UDP.recvfrom(buffersize)
    outdata = np.frombuffer(data,dtype=np.double)
    return outdata
    

if __name__ == "__main__":

    in_files = 'GRMN0119.MP4'
    model_path = './model/laneunet.pth'
    cap = cv2.VideoCapture(in_files)
    # out_files = get_output_filenames(args)

    h = parameter.imgh
    w = parameter.imgw

    #     #########################################
    # # Camera Pose
    # CameraPose = image2bev._DictObjHolder({
    #         "Pitch": parameter.pitch,
    #         "Yaw": parameter.yaw,
    #         "Roll": parameter.roll,
    #         "Height": parameter.height,
    #         })

    # # IntrinsicMatrix
    # mtx = parameter.mtx
    # dist = parameter.dist
    # IntrinsicMatrix = np.transpose(mtx)

    # # Out Image View
    # OutImageView = image2bev._DictObjHolder({
    #         "distAheadOfSensor": parameter.distAheadOfSensor,
    #         "spaceToLeftSide": parameter.spaceToLeftSide,
    #         "spaceToRightSide": parameter.spaceToRightSide,
    #         "bottomOffset": parameter.bottomOffset,
    #         })

    # OutImageSize = np.array([np.nan, np.int_(w/2)])  # image H, image W

    net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    w_frame = 512
    h_frame = 256
    # fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    # writer = cv2.VideoWriter('binarylane.avi',fourcc, 30, (w_frame,h_frame))
    
    # create Socket UDP
    socket_UDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    import time
    socket_UDP_receive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receive_addr = ("", 8080)
    socket_UDP_receive.bind(receive_addr)
    socket_UDP_receive.settimeout(1)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # img = cv2.undistort(img, mtx, dist, None, mtx)
            img_pil = Image.fromarray(img)
            mask = predict_img(net=net,                                            
                               full_img=img_pil,
                               scale_factor=1,
                               out_threshold=0.2,
                               device=device)
            # result = mask_to_image(mask)
                # plot_img_and_mask(img, mask)
            
            binaryimg = (mask*255).astype(np.uint8)
            binaryimg = cv2.resize(binaryimg, (w, h))
            binaryimg_128 = cv2.resize(binaryimg,(256, 128)).astype(np.uint8)
            
            UDP_msg = binaryimg_128.tobytes()
            
            UDP_send(socket_UDP=socket_UDP, server_address=server_address, msg=UDP_msg)
            
            # receive
            try:
                recv_data, addr = socket_UDP_receive.recvfrom(64)
                outdata = np.frombuffer(recv_data,dtype=np.single)
                print(outdata)
            except socket.timeout as e:
                pass
            
            # outdata = UDP_receive(socket_UDP=socket_UDP)
            lanemask, lane_coords = lanecluster.lane_mask_coords(cv2.resize(binaryimg,(512,256)))

            cv2.namedWindow('binaryimg',cv2.WINDOW_AUTOSIZE)
            cv2.imshow('binaryimg', cv2.resize(binaryimg,(512,256)))
            cv2.namedWindow('lanemask',cv2.WINDOW_AUTOSIZE)
            cv2.imshow('lanemask', lanemask)
 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cv2.destroyAllWindows()
    socket_UDP.close()
    socket_UDP_receive.close()
