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

from utils import parameter
from utils import image2bev
from utils import lane_parameter
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


if __name__ == "__main__":

    in_files = '/home/jetson/Videos/output.mp4'
    model_path = './model/laneunet.pth'
    cap = cv2.VideoCapture(in_files)
    # out_files = get_output_filenames(args)

    h = parameter.imgh
    w = parameter.imgw

        #########################################
    # Camera Pose
    CameraPose = image2bev.DictObjHolder({
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
    OutImageView = image2bev.DictObjHolder({
            "distAheadOfSensor": parameter.distAheadOfSensor,
            "spaceToLeftSide": parameter.spaceToLeftSide,
            "spaceToRightSide": parameter.spaceToRightSide,
            "bottomOffset": parameter.bottomOffset,
            })

    OutImageSize = np.array([np.nan, np.int_(w/2)])  # image H, image W

    net = UNet(n_channels=3, n_classes=1)
    
    

    logging.info("Loading model {}".format(model_path))
    
 
    with open( './model/pkl_1.4model.pth', 'rb') as f:
        info_dict = pkl.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    # net.load_state_dict(torch.load(info_dict, map_location=device))
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    # with open( 'pkl_1.4model.pth', 'rb') as f:
#     info_dict = pkl.load(f)

# model.load_state_dict(info_dict['model'])

    logging.info("Model loaded !")

    # for i, fn in enumerate(in_files):
    #     logging.info("\nPredicting image {} ...".format(fn))

    #     img = Image.open(fn)

    #     mask = predict_img(net=net,
    #                        full_img=img,
    #                        scale_factor=args.scale,
    #                        out_threshold=args.mask_threshold,
    #                        device=device)

    #     if not args.no_save:
    #         out_fn = out_files[i]
    #         result = mask_to_image(mask)
    #         result.save(out_files[i])

    #         logging.info("Mask saved to {}".format(out_files[i]))

    #     if args.viz:
    #         logging.info("Visualizing results for image {}, close to continue ...".format(fn))
    #         plot_img_and_mask(img, mask)
            
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
            
            frame = cv2.resize(frame,(w,h))

            out_image = cv2.addWeighted(frame,1,add_img,1,0.0)
            
            mittel_lane = (ego_right_lane + ego_left_lane) / 2
            mittel_mask_img = lane_parameter.drawmittellane(binaryimg, birdseyeviewimage, unwarp_matrix, mittel_lane)
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
            cv2.namedWindow('binaryresult',cv2.WINDOW_AUTOSIZE)
            cv2.imshow('binaryresult', binaryimg)            
            cv2.namedWindow('result',cv2.WINDOW_AUTOSIZE)
            cv2.imshow('result', out_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
