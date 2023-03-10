# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:44:50 2020

@author: gao
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.measure import regionprops
# from skimage.color import label2rgb

class DictObjHolder(object):
    def __init__(self, dct):
        self.dct = dct

    def __getattr__(self, name):
        return self.dct[name]
    
def rotX(a):
    a = np.deg2rad(a)
    R = np.array([[1,0,0],
                  [0,np.cos(a),-np.sin(a)],
                  [0,np.sin(a),np.cos(a)]])
    return R

def rotY(a):
    a = np.deg2rad(a)
    R = np.array([[np.cos(a),0,np.sin(a)],
                  [0,1,0],
                  [-np.sin(a),0,np.cos(a)]])
    return R

def rotZ(a):
    a = np.deg2rad(a)
    R = np.array([[np.cos(a),-np.sin(a),0],
                  [np.sin(a),np.cos(a),0],
                  [0,0,1]])
    return R 

class BirdsEyeView:
    def __init__(self):
        self.scalex = []    # scale image W x-direction
        self.scaley = []    # scale image H y-direction
        self.vehicleHomography = []
        self.BirdsEyeViewTransform = []
        self.worldHW = []
        self.bevSize = []
        self.unwarp_matrix = []
  
    def birdseyeviewimage(self,image,mtx,CameraPose,OutImgView,OutImgSize):
        Pitch = CameraPose.Pitch
        Yaw = CameraPose.Yaw
        Roll = CameraPose.Roll
        Height = CameraPose.Height
        
        distAheadOfSensor = OutImgView.distAheadOfSensor
        spaceToLeftSide = OutImgView.spaceToLeftSide 
        spaceToRightSide = OutImgView.spaceToRightSide
        bottomOffset = OutImgView.bottomOffset
        
        outView = np.array([bottomOffset,distAheadOfSensor,-spaceToLeftSide,spaceToRightSide])
        reqImgHW = OutImgSize.copy()
        self.worldHW  = np.abs([outView[1]-outView[0], outView[3]-outView[2]])
        
        rotation = np.linalg.multi_dot([rotY(180),rotZ(-90),rotZ(-Yaw),rotX(90-Pitch),rotZ(Roll)])
        rotationMatrix = np.linalg.multi_dot([rotZ(-Yaw),rotX(90-Pitch),rotZ(Roll)])
        sl = [0,0]
        translationInWorldUnits = [sl[1], sl[0], Height]
        translation = np.dot(translationInWorldUnits,rotationMatrix)
        camMatrix = np.dot(np.vstack([rotation,translation]),mtx.T)
        tform2D = np.array([camMatrix[0,:], camMatrix[1,:], camMatrix[3,:]])
        ImageToVehicleTransform = np.linalg.inv(tform2D)
        self.vehicleHomography = ImageToVehicleTransform
        adjTform = np.array([[0, -1, 0],
                             [-1, 0, 0],
                             [0, 0, 1]])
        bevTform = np.dot(self.vehicleHomography,adjTform)
        
        nanIdxHW = np.isnan(reqImgHW)
        if ~nanIdxHW.any():
            scaleXY = np.flipud((reqImgHW-1)/self.worldHW)
            outSize = reqImgHW
            self.scalex = scaleXY[0]
            self.scaley = scaleXY[1]
        else:
            scale   = (reqImgHW[~nanIdxHW]-1)/self.worldHW[~nanIdxHW]
            scaleXY = np.hstack([scale, scale])
            worldDim = self.worldHW[nanIdxHW]
            outDimFrac = scale*worldDim
            outDim     = np.round(outDimFrac)+1
            outSize = reqImgHW
            outSize[nanIdxHW] = outDim
            self.scalex = scaleXY[0]
            self.scaley = scaleXY[1]
        OutputView = outView
        dYdXVehicle = np.array([OutputView[3], OutputView[1]])
        tXY         = scaleXY*dYdXVehicle
        viewMatrix = np.array([[scaleXY[0], 0, 0],
                               [0, scaleXY[1], 0],
                               [tXY[0]+1, tXY[1]+1, 1]])
        self.BirdsEyeViewTransform = np.transpose(np.dot(bevTform, viewMatrix))
        self.bevSize = np.int_(np.flipud(outSize))
        birdsEyeViewImage = cv2.warpPerspective(image,self.BirdsEyeViewTransform,tuple(np.int_(np.flipud(outSize))))
        self.unwarp_matrix = np.linalg.inv(self.BirdsEyeViewTransform)/self.scalex
        return birdsEyeViewImage

    def imagetovehicle(self, imgpoint):
        # image pixel point (image coordination) to real world point position (vehicle coordination)
        
        worldpoint = np.dot(self.BirdsEyeViewTransform, np.hstack([imgpoint, 1]))
        worldpoint /= worldpoint[2]
        
        worldpoint_sc = worldpoint / np.array([self.scalex, self.scaley, 1])
        vehicle_Matrix = np.array([[0, -1, self.worldHW[0]],
                                   [-1, 0, self.worldHW[1]/2],
                                   [0, 0, 1]])
        worldpoint_vc = np.dot(vehicle_Matrix, worldpoint_sc)
        point_worldcoor = worldpoint_vc[0:2]
        point_bevcoor = np.int_(np.round(worldpoint[0:2]))
        return point_worldcoor, point_bevcoor
    
    def bevimagetovehicle(self, bevpoint):
        # image pixel point (image coordination) to real world point position (vehicle coordination)
        
        worldpoint = np.hstack([bevpoint, 1])
        
        worldpoint_sc = worldpoint / np.array([self.scalex, self.scaley, 1])
        vehicle_Matrix = np.array([[1, 0, -self.worldHW[1]/2],
                                    [0, 1, 0],
                                    [0, 0, 1]])
        worldpoint_vc = np.dot(vehicle_Matrix, worldpoint_sc)
        point_worldcoor = worldpoint_vc[0:2]
        return point_worldcoor
    
    def vehicletoimage(self, worldpoint_vc):
        # real world point (vehicle coordination) to image pixel point (image coordination) 
        # and bird eye view image point (bev coordination)
        
        vehicle_Matrixinv = np.array([[0, -1, self.worldHW[1]/2],
                                      [-1, 0, self.worldHW[0]],
                                      [0, 0, 1]])
        worldpoint = np.dot(vehicle_Matrixinv, np.hstack([worldpoint_vc,1]))*np.array([self.scalex, self.scaley, 1])
        imgpoint = np.dot(np.linalg.inv(self.BirdsEyeViewTransform),worldpoint)
        imgpoint /= imgpoint[2]
        point_imgcoor = np.int_(np.round(imgpoint[0:2]))
        # point_bevcoor = np.int_(np.round(worldpoint[0:2]))
        return point_imgcoor
    
def bev_perspective(image, mtx, CameraPose, OutImageView, OutImageSize):
    
    birdseyeview = BirdsEyeView()
    BirdseyeviewImage = birdseyeview.birdseyeviewimage(image,mtx,CameraPose,OutImageView,OutImageSize)
    # plt.imshow(BirdseyeviewImage)
    # BirdseyeviewImage = cv2.resize(BirdseyeviewImage,(512,256))
    unwarp_matrix = birdseyeview.unwarp_matrix
    return BirdseyeviewImage, unwarp_matrix, birdseyeview

def lanelabel(binary_image, min_area_threshold=500):
    #ret, binary_image = cv2.threshold(binary_image, 1, 255,cv2.THRESH_BINARY)
    label_image = label(binary_image)
    region_props = regionprops(label_image)
    lane_label_ret = []
    for region in region_props:
        if region.area > min_area_threshold:
            lane_label_ret.append(region)
    return lane_label_ret

def left_right_lane(fit_params, warpimage):
    x_th = warpimage.shape[1]/2
    left_lane = []
    right_lane = []
    right_dist = []
    left_dist = []
    for fit_param in fit_params:
        if fit_param[2] >= x_th:
            right_lane.append(fit_param)
            right_dist.append(fit_param[2])
        else:
            left_lane.append(fit_param)
            left_dist.append(fit_param[2])
    
    ego_right_index = right_dist.index(min(right_dist))
    ego_left_index = left_dist.index(min(left_dist))
    ego_right_lane = right_lane[ego_right_index]
    ego_left_lane = left_lane[ego_left_index]
    return ego_right_lane,ego_left_lane

def isgood(fit_param):
    a = fit_param[0]
    b = fit_param[1]
    isGood = abs(a) < 0.003 and abs(b) < 0.8
    return isGood

def lanefit(binary_image,mtx,CameraPose, OutImageView, OutImageSize):   
    warpimage, unwarp_matrix, birdseyeview = bev_perspective(binary_image.astype(np.uint8), mtx, CameraPose, OutImageView, OutImageSize)
    warpimage[warpimage[:,:]!=0]=255
    lane_label = lanelabel(warpimage, min_area_threshold=300)
    [binary_img_H, binary_img_W] = binary_image.shape
    
    fit_params = []

    init_fit_params = np.array([[-8.59772640e-07,  2.72346164e-02,  2.04014669e+02],
                     [ 6.64430504e-06, -1.32049055e-02,  6.38566362e+02]])
    
    way_plot_y = np.linspace(0, warpimage.shape[0]-1, OutImageView.distAheadOfSensor)
    line_img = np.zeros_like(warpimage).astype(np.uint8)
    if lane_label is None:
        init_mittel_lane = (init_fit_params[1] + init_fit_params[0]) / 2
        fit_x = init_mittel_lane[0] * way_plot_y ** 2 + init_mittel_lane[1] * way_plot_y ** 1 + init_mittel_lane[2]
        idx_fitx = (np.int_(fit_x)>=0) & (np.int_(fit_x)<=warpimage.shape[1]-1)
        warp_y = np.int_(way_plot_y)[idx_fitx]
        warp_x = np.int_(fit_x)[idx_fitx]
        way_pts = np.vstack((warp_x,warp_y))
        worldpoints = []
        for i in range(way_pts.shape[1]):   
            worldpoint = birdseyeview.bevimagetovehicle(way_pts[:,i])
            worldpoints.append(worldpoint)
        ret = {
                'fit_params': init_fit_params,
                'ego_right_lane': init_fit_params[1],
                'ego_left_lane': init_fit_params[0],
                'mittel_lane': init_mittel_lane,
                'waypoints': worldpoints,
                'laneimg': line_img,
                }
        return ret
    
    try: 
        #line_img = np.zeros_like(warpimage).astype(np.uint8)
        for i in range(len(lane_label)):
            nonzero_y = lane_label[i].coords[:,0]
            nonzero_x = lane_label[i].coords[:,1]        
            fit_param = np.polyfit(nonzero_y, nonzero_x, 2) # np.polyfit(nonzero_y, nonzero_x, 3)
            isGood = isgood(fit_param)
            if isGood == True:
                fit_params.append(fit_param)
                
                plot_y = np.linspace(0, warpimage.shape[0]-1, warpimage.shape[0])
                fit_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y ** 1 + fit_param[2]    
                idx_fitx = (np.int_(fit_x)>=0) & (np.int_(fit_x)<=warpimage.shape[1]-1)
                warp_y = np.int_(plot_y)[idx_fitx]
                warp_x = np.int_(fit_x)[idx_fitx]
                warp_ys.extend(warp_y)
                warp_xs.extend(warp_x)
            
        line_pts = (warp_ys, warp_xs)   
        line_img[line_pts] = 255
        # mask_img = cv2.warpPerspective(line_img, unwarp_matrix, (binary_image.shape[1], binary_image.shape[0]))
        # src_x = np.array(mask_img.nonzero()[1])
        # src_y = np.array(mask_img.nonzero()[0])
        # src_xy = np.transpose(np.vstack((src_x,src_y)))
        # points_xy = [tuple(x) for x in src_xy]
        # lane_image = np.zeros_like(binary_image).astype(np.uint8)
        # for points in points_xy:
        #     lane_image = cv2.circle(lane_image,points,3,255,-1)
        ego_right_lane,ego_left_lane = left_right_lane(fit_params, warpimage)
        mittel_lane = (ego_right_lane + ego_left_lane) / 2
        fit_x = mittel_lane[0] * way_plot_y ** 2 + mittel_lane[1] * way_plot_y ** 1 + mittel_lane[2]
        idx_fitx = (np.int_(fit_x)>=0) & (np.int_(fit_x)<=warpimage.shape[1]-1)
        warp_y = np.int_(way_plot_y)[idx_fitx]
        warp_x = np.int_(fit_x)[idx_fitx]
        way_pts = np.vstack((warp_x,warp_y))
        worldpoints = []
        for i in range(way_pts.shape[1]):   
            worldpoint = birdseyeview.bevimagetovehicle(way_pts[:,i])
            worldpoints.append(worldpoint)
        ret = {
                'fit_params': fit_params,
                'ego_right_lane': ego_right_lane,
                'ego_left_lane': ego_left_lane,
                'mittel_lane': mittel_lane,
                'waypoints': worldpoints,
                'laneimg': line_img,
                }
        return ret
    except:
        init_mittel_lane = (init_fit_params[1] + init_fit_params[0]) / 2
        fit_x = init_mittel_lane[0] * way_plot_y ** 2 + init_mittel_lane[1] * way_plot_y ** 1 + init_mittel_lane[2]
        idx_fitx = (np.int_(fit_x)>=0) & (np.int_(fit_x)<=warpimage.shape[1]-1)
        warp_y = np.int_(way_plot_y)[idx_fitx]
        warp_x = np.int_(fit_x)[idx_fitx]
        way_pts = np.vstack((warp_x,warp_y))
        worldpoints = []
        for i in range(way_pts.shape[1]):   
            worldpoint = birdseyeview.bevimagetovehicle(way_pts[:,i])
            worldpoints.append(worldpoint)
        ret = {
                'fit_params': init_fit_params,
                'ego_right_lane': init_fit_params[1],
                'ego_left_lane': init_fit_params[0],
                'mittel_lane': init_mittel_lane,
                'waypoints': worldpoints,
                'laneimg': line_img,
                }
        return ret

def drawlane(binary_image, warpimage, unwarp_matrix, ego_right_lane, ego_left_lane):   
    line_img = np.zeros_like(warpimage).astype(np.uint8)
    plot_y = np.linspace(0, warpimage.shape[0]-1, warpimage.shape[0])
    fit_params=[]
    warp_xs = []
    warp_ys = []
    fit_params.append((ego_right_lane, ego_left_lane))
    
    for fit_param in fit_params[0]:
          fit_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y ** 1 + fit_param[2]    
          idx_fitx = (np.int_(fit_x)>=0) & (np.int_(fit_x)<=warpimage.shape[1]-1)
          warp_y = np.int_(plot_y)[idx_fitx]
          warp_x = np.int_(fit_x)[idx_fitx]
          warp_ys.extend(warp_y)
          warp_xs.extend(warp_x)
    line_pts = (warp_ys, warp_xs)   
    line_img[line_pts] = 255
    mask_img = cv2.warpPerspective(line_img, unwarp_matrix, (binary_image.shape[1], binary_image.shape[0]))
    # src_x = np.array(mask_img.nonzero()[1])
    # src_y = np.array(mask_img.nonzero()[0])
    # src_xy = np.transpose(np.vstack((src_x,src_y)))
    src_xy = np.argwhere(mask_img != 0)
    src_xy = np.flip(src_xy[:,0:2],1)
    points_xy = [tuple(x) for x in src_xy]
    lane_image = np.zeros_like(binary_image).astype(np.uint8)
    for points in points_xy:
        lane_image = cv2.circle(lane_image,points,3,255,-1)
    return lane_image

def drawmittellane(binary_image, warpimage, unwarp_matrix, mittel_lane):   
    line_img = np.zeros_like(warpimage).astype(np.uint8)
    plot_y = np.linspace(0, warpimage.shape[0]-1, warpimage.shape[0])
    fit_param = mittel_lane
    fit_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y ** 1 + fit_param[2]    
    idx_fitx = (np.int_(fit_x)>=0) & (np.int_(fit_x)<=warpimage.shape[1]-1)
    warp_y = np.int_(plot_y)[idx_fitx]
    warp_x = np.int_(fit_x)[idx_fitx]
    line_pts = (warp_y, warp_x)   
    line_img[line_pts] = 255
    mask_img = cv2.warpPerspective(line_img, unwarp_matrix, (binary_image.shape[1], binary_image.shape[0]))
    # src_x = np.array(mask_img.nonzero()[1])
    # src_y = np.array(mask_img.nonzero()[0])
    # src_xy = np.transpose(np.vstack((src_x,src_y)))
    src_xy = np.argwhere(mask_img != 0)
    src_xy = np.flip(src_xy[:,0:2],1)
    points_xy = [tuple(x) for x in src_xy]
    lane_image = np.zeros_like(binary_image).astype(np.uint8)
    for points in points_xy:
        lane_image = cv2.circle(lane_image,points,3,255,-1)
    return lane_image
    
def imageadd(img1,img2):
    addimage = cv2.addWeighted(img1,1,img2,1,0)
    return addimage


"""
lane_label = lanelabel(binary_image, min_area_threshold=300)
line_img = np.zeros_like(binary_image).astype(np.uint8)
for lane in lane_label:
    nonzero_y = lane.coords[:,0]
    nonzero_x = lane.coords[:,1]
    line_pts = (nonzero_y, nonzero_x)
    line_img[line_pts]=255
    
label_image = label(line_img)
label_overlay = label2rgb(label_image, image=None, bg_label=0)
resize_label_image = (cv2.resize(label_overlay,(1920,1080))*255).astype(np.uint8)
image_label_overlay = label2rgb(resize_label_image, image=image, bg_label=0)
    """
