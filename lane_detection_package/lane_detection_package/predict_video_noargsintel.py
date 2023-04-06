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
from utils.lane_parameter import DictObjHolder, bev_perspective, lanefit, drawlane, drawmittellane, PolynomialRegression, drawsinglelane, get_fit_param, insertLaneBoundary
from utils.lanecluster import lane_mask_coords
import pickle as pkl
import collections


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

def moving_average(array, window_data):
    window_data[:-1] = window_data[1:]
    window_data[-1] = array
    moving_avg = np.mean(window_data, axis=0)

    return moving_avg

mtx = np.array([[633.0128, 0., 425.0031],
                [0., 635.3088, 228.2753],
                [0.,0.,1.]
                ])
dist = np.array([0.1020, -0.1315, 0, 0, 0])

pitch = 2
yaw = 0
roll = 0
height = 1

distAheadOfSensor = 30
spaceToLeftSide = 5    
spaceToRightSide = 5
bottomOffset = 1

imgw = 848
imgh = 480
window_size = 10

if __name__ == "__main__":

    in_files = '/home/fmon005/Videos/test_campus1.mp4'
    model_path = './model/Lanenet0304.pth'
    # CarlaCP_epoch20.pth
    # old_CP_epoch14.pth
    cap = cv2.VideoCapture(in_files)
    # out_files = get_output_filenames(args)
    legend_img = cv2.imread('Legend.png')
    l_h, l_w, l_c = legend_img.shape

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

    left_fit_model = RANSACRegressor(PolynomialRegression(degree=2), residual_threshold=10)
    right_fit_model = RANSACRegressor(PolynomialRegression(degree=2), residual_threshold=10)
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

    fourcc=cv2.VideoWriter_fourcc(*'XVID')
    # videoWriter=cv2.VideoWriter('nolegendnachtrainsplines.avi',fourcc,15,(imgw,imgh))

    left_window_data = np.zeros((window_size,3))
    right_window_data = np.zeros((window_size,3))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            #img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            # birdseyeviewimage, unwarp_matrix, birdseyeview = bev_perspective(img, mtx, CameraPose, OutImageView, OutImageSize)
            
            warpimage, unwarp_matrix, birdseyeview = bev_perspective(binaryimg_original.astype(np.uint8), mtx, CameraPose, OutImageView, OutImageSize)
            
            imageX, imageY = np.where(warpimage)
            xyBoundaryPoints = birdseyeview.bevimagetovehicle(np.column_stack((imageY,imageX)))

            leftlane = xyBoundaryPoints[xyBoundaryPoints[:,1]>0]
            rightlane = xyBoundaryPoints[xyBoundaryPoints[:,1]<=0]

            left_init_param = np.array([])
            right_init_param = np.array([])
            leftparam = get_fit_param(leftlane, left_init_param, left_fit_model)
            rightparam = get_fit_param(rightlane, right_init_param, right_fit_model)

            left_lane_img = insertLaneBoundary(img, warpimage, leftparam, OutImageView, birdseyeview)
            lane_img = insertLaneBoundary(left_lane_img, warpimage, rightparam, OutImageView, birdseyeview)



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
            """
            lane_fit_params = lanefit(binaryimg_original,mtx,CameraPose, OutImageView, OutImageSize, fit_model)
            ego_right_lane = lane_fit_params['ego_right_lane']
            ego_left_lane = lane_fit_params['ego_left_lane']
            # waypoints = lane_fit_params['waypoints']
            #line_img = lane_fit_params['laneimg']
            # print(ego_left_lane)
            # print(ego_right_lane)
            # print('-------------')

            if ego_left_lane.size > 0:
                ego_left_lane = moving_average(ego_left_lane, left_window_data)
                left_window_data = np.vstack((left_window_data[1:], ego_left_lane))
            if ego_right_lane.size > 0:
                ego_right_lane = moving_average(ego_right_lane, right_window_data)
                right_window_data = np.vstack((right_window_data[1:], ego_right_lane))

            # mask_img = drawlane(binaryimg_original, birdseyeviewimage, unwarp_matrix, ego_right_lane, ego_left_lane)
            
            mask_img_right = drawsinglelane(binaryimg_original, birdseyeviewimage, unwarp_matrix, ego_right_lane)
            mask_img_left = drawsinglelane(binaryimg_original, birdseyeviewimage, unwarp_matrix, ego_left_lane)

            g_b = np.zeros_like(mask_img_right).astype(np.uint8)
            # add_img_l = cv2.merge((g_b, mask_img_right, mask_img_left))
            
            mask_right_rgb = np.zeros_like(img).astype(np.uint8)
            mask_right_rgb[mask_img_right==255,:]=[0,255,255]
            mask_left_rgb = np.zeros_like(img).astype(np.uint8)
            mask_left_rgb[mask_img_left==255,:]=[0,255,0]
            add_img_l = cv2.addWeighted(mask_right_rgb,1.0, mask_left_rgb, 1.0,0)
            
            # add_img_l = yellow_img+green_img

            frame = cv2.resize(frame,(imgw,imgh))



            # out_image = cv2.addWeighted(frame,1,add_img_l,1,0.0)
            
            # mittel_lane = (ego_right_lane + ego_left_lane) / 2
            # mittel_mask_img = drawmittellane(binaryimg_original, birdseyeviewimage, unwarp_matrix, mittel_lane)
            # add_img1 = cv2.merge((g_b, g_b,mittel_mask_img))

            # out_image = cv2.addWeighted(out_image,1,add_img1,1,0.0)
            
            #print('waypoints: ', waypoints)
    
            # result = (mask*255).astype(np.uint8)
            
            # result_resize = cv2.resize(result,(512,256))
            g_r = np.zeros_like(binaryimg_original).astype(np.uint8)
            
            add_img = cv2.merge((g_r, g_r, binaryimg_original))
            # img_resize = cv2.resize(img,(512,256))
            
            overlayimg = cv2.addWeighted(img,1,add_img,1,0.0)
            # out_img = cv2.cvtColor(overlayimg, cv2.COLOR_RGB2BGR)
            out_image = cv2.addWeighted(overlayimg,1.0,add_img_l,1.0,0.0)
            
            # out_image[0:l_h, imgw-l_w:imgw] = legend_img
            
            if ii>73:
                videoWriter.write(out_image)

            ii=ii+1
            cv2.namedWindow('out_img',cv2.WINDOW_AUTOSIZE)
            cv2.imshow('out_img', overlayimg)
            cv2.namedWindow('img',cv2.WINDOW_AUTOSIZE)
            cv2.imshow('img', warpimage)  
            cv2.namedWindow('birdseyeviewimage',cv2.WINDOW_AUTOSIZE)
            cv2.imshow('birdseyeviewimage', birdseyeviewimage)          
            cv2.namedWindow('result',cv2.WINDOW_AUTOSIZE)
            cv2.imshow('result', out_image)
            """
            cv2.namedWindow('result',cv2.WINDOW_AUTOSIZE)
            cv2.imshow('result', lane_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    # videoWriter.release()
