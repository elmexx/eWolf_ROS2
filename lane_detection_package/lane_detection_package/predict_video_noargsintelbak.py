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

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset

# from utils import parameter
# from utils import image2bev
from utils.lane_parameter import DictObjHolder, bev_perspective, lanefit, drawlane, drawmittellane
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

pitch = -1
yaw = 0
roll = 0
height = 1

distAheadOfSensor = 20
spaceToLeftSide = 4    
spaceToRightSide = 4
bottomOffset = 1

imgw = 848
imgh = 480

if __name__ == "__main__":

    in_files = '/home/jetson/Videos/output.mp4'
    model_path = './model/CP_epoch8.pth'
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

    net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    # net.load_state_dict(torch.load(info_dict, map_location=device))
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()


    logging.info("Model loaded !")


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
            
            # Bird's Eye View Image
            birdseyeviewimage, unwarp_matrix, birdseyeview = bev_perspective(img, mtx, CameraPose, OutImageView, OutImageSize)
            
            warpimage, unwarp_matrix, birdseyeview = bev_perspective(binaryimg_original.astype(np.uint8), mtx, CameraPose, OutImageView, OutImageSize)
            
            # lane parameter in vehicle coordination
            lane_fit_params = lanefit(binaryimg_original,mtx,CameraPose, OutImageView, OutImageSize)
            ego_right_lane = lane_fit_params['ego_right_lane']
            ego_left_lane = lane_fit_params['ego_left_lane']
            # waypoints = lane_fit_params['waypoints']
            line_img = lane_fit_params['laneimg']
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
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
