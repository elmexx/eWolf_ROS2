# -*- coding: utf-8 -*-

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset

import parameter
import image2bev
import lane_parameter

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

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
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
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

def plot_mask_to_img(img, mask_img):
    red_img = Image.new('RGB', img.size, (255,0,0))
    mask_img = mask_img.resize(img.size)
    im = Image.composite(red_img, img, mask_img)
    im.show()
    return im


if __name__ == "__main__":
    
    image_path = '20200923162916_00060.png'
    model_path = './checkpoints/CP_epoch8.pth'


    net = UNet(n_channels=3, n_classes=1)
    
    logging.info("Loading model {}".format(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(model_path, map_location=device))

    logging.info("Model loaded !")
    
    logging.info("\nPredicting image ...")
    
    img = cv2.imread(image_path)
    
    h, w, c = img.shape

        #########################################
    # Camera Pose
    CameraPose = image2bev._DictObjHolder({
            "Pitch": parameter.pitch,
            "Yaw": parameter.yaw,
            "Roll": parameter.roll,
            "Height": parameter.height,
            })

    # IntrinsicMatrix
    mtx = parameter.mtx
    dist = parameter.dist
    IntrinsicMatrix = np.transpose(mtx)

    # Out Image View
    OutImageView = image2bev._DictObjHolder({
            "distAheadOfSensor": parameter.distAheadOfSensor,
            "spaceToLeftSide": parameter.spaceToLeftSide,
            "spaceToRightSide": parameter.spaceToRightSide,
            "bottomOffset": parameter.bottomOffset,
            })

    OutImageSize = np.array([np.nan, np.int_(w/2)])  # image H, image W


    img = cv2.undistort(img, mtx, dist, None, mtx)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    # img_pil = Image.open(image_path)
    mask = predict_img(net=net,
                   full_img=img_pil,
                   scale_factor=1,
                   out_threshold=0.4,
                   device=device)
    # plot_img_and_mask(img, mask)
    binaryimg = (mask*255).astype(np.uint8)
    binaryimg = cv2.resize(binaryimg, (w, h))
    
    # Bird's Eye View Image
    birdseyeviewimage, unwarp_matrix, birdseyeview = lane_parameter.bev_perspective(img, IntrinsicMatrix, CameraPose, OutImageView, OutImageSize)
    
    # lane parameter in vehicle coordination
    lane_fit_params = lane_parameter.lanefit(binaryimg,IntrinsicMatrix,CameraPose, OutImageView, OutImageSize)
    ego_right_lane = lane_fit_params['ego_right_lane']
    ego_left_lane = lane_fit_params['ego_left_lane']
    waypoints = lane_fit_params['waypoints']

    mask_img = lane_parameter.drawlane(binaryimg, birdseyeviewimage, unwarp_matrix, ego_right_lane, ego_left_lane)
    
    g_b = np.zeros_like(mask_img).astype(np.uint8)
    add_img = cv2.merge((mask_img,g_b, g_b))
    out_image = cv2.addWeighted(img,1,add_img,1,0.0)
    
    mittel_lane = (ego_right_lane + ego_left_lane) / 2
    mittel_mask_img = lane_parameter.drawmittellane(binaryimg, birdseyeviewimage, unwarp_matrix, mittel_lane)
    add_img1 = cv2.merge((g_b, g_b,mittel_mask_img))
    out_image = cv2.addWeighted(out_image,1,add_img1,1,0.0)
    
    print('waypoints: ', waypoints)

