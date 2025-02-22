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
from sklearn.linear_model import RANSACRegressor

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset

# from utils import parameter
# from utils import image2bev
from utils.lane_parameter import DictObjHolder, bev_perspective, lanefit, drawlane, drawmittellane, PolynomialRegression
from utils.lanecluster import lane_mask_coords
import pickle as pkl
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
    parser.add_argument('--model', '-m', default='pkl_1.4model.pth',
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


mtx = np.array([[633.0128, 0., 425.0031],
                [0., 635.3088, 228.2753],
                [0.,0.,1.]
                ])
dist = np.array([0.1020, -0.1315, 0, 0, 0])

pitch = 3
yaw = 3
roll = 0
height = 1.6

distAheadOfSensor = 30
spaceToLeftSide = 5    
spaceToRightSide = 5
bottomOffset = 1

imgw = 848
imgh = 480

if __name__ == "__main__":

    in_files = '/home/fmon005/Videos/output1.mp4'
    
    cap = cv2.VideoCapture(in_files)
    # out_files = get_output_filenames(args)


        #########################################
    # Camera Pose
    CameraPose = DictObjHolder({
                "Pitch": pitch,
                "Yaw": yaw,
                "Roll": roll,
                "Height": height,
            })

    # IntrinsicMatrix

    # IntrinsicMatrix = np.transpose(mtx)

    # Out Image View
    OutImageView = DictObjHolder({
                "distAheadOfSensor": distAheadOfSensor,
                "spaceToLeftSide": spaceToLeftSide,
                "spaceToRightSide": spaceToRightSide,
                "bottomOffset": bottomOffset,
            })

    OutImageSize = np.array([np.nan, np.int_(imgw/2)])  # image H, image W

    fit_model = RANSACRegressor(PolynomialRegression(degree=2), residual_threshold=10, random_state=0)
    model_path = './model/Lanenet.pth'
    net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    # net.load_state_dict(torch.load(info_dict, map_location=device))
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()


    logging.info("Model loaded !")
    
    ii = 1
    img_path = '/home/fmon005/Pictures/00153427_rgb.png'
    frame = cv2.imread(img_path)
    img = cv2.resize(frame, (imgw,imgh))
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
    binaryimg_original = cv2.resize(binaryimg, (imgw, imgh))
    binaryimg_256 = cv2.resize(binaryimg,(512,256))
    
    # Bird's Eye View Image
    birdseyeviewimage, unwarp_matrix, birdseyeview = bev_perspective(img, mtx, CameraPose, OutImageView, OutImageSize)
    
    warpimage, unwarp_matrix, birdseyeview = bev_perspective(binaryimg_original.astype(np.uint8), mtx, CameraPose, OutImageView, OutImageSize)
    
    # lanemask, lane_coords = lane_mask_coords(warpimage)
    # fit_params = []
    # for i in range(len(lane_coords)):
    #     nonzero_y = lane_coords[i][:,0]
    #     nonzero_x = lane_coords[i][:,1]        
    #     fit_param = np.polyfit(nonzero_y, nonzero_x, 2)
    #     fit_params.append(fit_param)
    
    # line_img = np.zeros_like(warpimage).astype(np.uint8)
    # plot_y = np.linspace(150, warpimage.shape[0]-1, warpimage.shape[0])
    # p_xs = []
    # p_ys = []
    
    # for fit_param in fit_params:
    #     fit_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y  + fit_param[2]
    #     idx_fitx = (np.int_(fit_x)>=0) & (np.int_(fit_x)<=warpimage.shape[1]-1)
    #     p_y = np.int_(plot_y)[idx_fitx]
    #     p_x = np.int_(fit_x)[idx_fitx]
    #     p_ys.extend(p_y)
    #     p_xs.extend(p_x)
    
    # line_pts = (p_ys, p_xs)
    # line_img[line_pts] = 255
    # src_xy = np.argwhere(line_img != 0)
    # src_xy = np.flip(src_xy[:,0:2],1)
    # points_xy = [tuple(x) for x in src_xy]
    # lane_image = np.zeros_like(warpimage).astype(np.uint8)
    # for points in points_xy:
    #     lane_image = cv2.circle(lane_image,points,3,255,-1)
    # g_bb = np.zeros_like(lane_image).astype(np.uint8)
    # add_imgb = cv2.merge((lane_image,g_bb, g_bb))
    # out_image_new = cv2.addWeighted(cv2.resize(frame,(512,256)),1,add_imgb,1,0.0)

    # lane parameter in vehicle coordination
    
    lane_fit_params = lanefit(binaryimg_original,mtx,CameraPose, OutImageView, OutImageSize, fit_model)
    ego_right_lane = lane_fit_params['ego_right_lane']
    ego_left_lane = lane_fit_params['ego_left_lane']
    # waypoints = lane_fit_params['waypoints']
    #line_img = lane_fit_params['laneimg']
    print(ego_left_lane)
    print(ego_right_lane)

    mask_img = drawlane(binaryimg_original, birdseyeviewimage, unwarp_matrix, ego_right_lane, ego_left_lane)
    
    g_b = np.zeros_like(mask_img).astype(np.uint8)
    add_img = cv2.merge((mask_img,g_b, g_b))
    
    frame = cv2.resize(frame,(imgw,imgh))

    out_image = cv2.addWeighted(frame,1,add_img,1,0.0)
    
    mittel_lane = (ego_right_lane + ego_left_lane) / 2
    mittel_mask_img = drawmittellane(binaryimg_original, birdseyeviewimage, unwarp_matrix, mittel_lane)
    add_img1 = cv2.merge((g_b, g_b,mittel_mask_img))

    out_image = cv2.addWeighted(out_image,1,add_img1,1,0.0)
    
    #print('waypoints: ', waypoints)

    # result = (mask*255).astype(np.uint8)
    
    # result_resize = cv2.resize(result,(512,256))
    # g_r = np.zeros_like(result_resize).astype(np.uint8)
    
    # add_img = cv2.merge((result_resize,g_r,g_r))
    # img_resize = cv2.resize(img,(512,256))
    
    # out_image = cv2.addWeighted(img_resize,1,add_img,1,0.0)
    # out_img = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
    
    # cv2.namedWindow('img',cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('img', img_resize)
    cv2.namedWindow('img',cv2.WINDOW_AUTOSIZE)
    cv2.imshow('img', warpimage)  
    cv2.namedWindow('birdseyeviewimage',cv2.WINDOW_AUTOSIZE)
    cv2.imshow('birdseyeviewimage', birdseyeviewimage)          
    cv2.namedWindow('result',cv2.WINDOW_AUTOSIZE)
    cv2.imshow('result', out_image)


cv2.destroyAllWindows()
