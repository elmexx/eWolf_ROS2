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
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RANSACRegressor
from numba import jit

class PolynomialRegression(object):
    def __init__(self, degree=2, coeffs=None):
        self.degree = degree
        self.coeffs = coeffs

    def fit(self, X, y):
        self.coeffs = np.polyfit(X.ravel(), y, self.degree)

    def get_params(self, deep=False):
        return {'coeffs': self.coeffs}

    def set_params(self, coeffs=None, random_state=None):
        self.coeffs = coeffs

    def predict(self, X):
        poly_eqn = np.poly1d(self.coeffs)
        y_hat = poly_eqn(X.ravel())
        return y_hat

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))

class DictObjHolder(object):
    def __init__(self, dct):
        self.dct = dct

    def __getattr__(self, name):
        return self.dct[name]
 
@jit 
def rotX(a):
    a = np.deg2rad(a)
    r = np.array([[1.0,0.0,0.0],
                  [0.0,np.cos(a),-np.sin(a)],
                  [0.0,np.sin(a),np.cos(a)]])
    return r

@jit
def rotY(a):
    a = np.deg2rad(a)
    r = np.array([[np.cos(a),0.0,np.sin(a)],
                  [0.0,1.0,0.0],
                  [-np.sin(a),0.0,np.cos(a)]])
    return r

@jit
def rotZ(a):
    a = np.deg2rad(a)
    r = np.array([[np.cos(a),-np.sin(a),0.0],
                  [np.sin(a),np.cos(a),0.0],
                  [0.0,0.0,1.0]])
    return r 

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
        
        self.distAheadOfSensor = OutImgView.distAheadOfSensor
        spaceToLeftSide = OutImgView.spaceToLeftSide 
        spaceToRightSide = OutImgView.spaceToRightSide
        bottomOffset = OutImgView.bottomOffset
        
        outView = np.array([bottomOffset,self.distAheadOfSensor,-spaceToLeftSide,spaceToRightSide])
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
        imgpoint_n = np.ones((imgpoint.shape[0],imgpoint.shape[1]+1))
        imgpoint_n[:,:-1] = imgpoint
        worldpoint = np.dot(imgpoint_n, self.BirdsEyeViewTransform.T) #np.hstack([imgpoint, 1]))
        last_col = worldpoint[:,-1]
        worldpoint /= last_col[:,None]
        
        worldpoint_sc = worldpoint / np.array([self.scalex, self.scaley, 1])
        vehicle_Matrix = np.array([[0, -1, self.distAheadOfSensor],
                                   [-1, 0, self.worldHW[1]/2],
                                   [0, 0, 1]])
        worldpoint_vc = np.dot(worldpoint_sc, vehicle_Matrix.T)
        point_worldcoor = worldpoint_vc[:,0:2]
        point_bevcoor = np.int_(np.round(worldpoint[:,0:2]))
        point_worldcoor = point_worldcoor[point_worldcoor[:,0]<=self.distAheadOfSensor]
        return point_worldcoor, point_bevcoor
    
    def bevimagetovehicle(self, bevpoint):

        worldpoint = np.ones((bevpoint.shape[0],bevpoint.shape[1]+1))
        worldpoint[:,:-1] = bevpoint
        
        worldpoint_sc = worldpoint / np.array([self.scalex, self.scaley, 1])

        vehicle_Matrix = np.array([[0, -1, self.distAheadOfSensor],
                                   [-1, 0, self.worldHW[1]/2],
                                   [0, 0, 1]])
        
        worldpoint_vc = np.dot(worldpoint_sc, vehicle_Matrix.T)
        point_worldcoor = worldpoint_vc[:,0:2]
        return point_worldcoor
    
    def vehicletoimage(self, worldpoint_vc):
        # real world point (vehicle coordination) to image pixel point (image coordination) 
        # and bird eye view image point (bev coordination)
        
        vehicle_Matrixinv = np.array([[0, -1, self.worldHW[1]/2],
                                      [-1, 0, self.worldHW[0]],
                                      [0, 0, 1]])
        worldpoint_n = np.ones((worldpoint_vc.shape[0],worldpoint_vc.shape[1]+1))
        worldpoint_n[:,:-1] = worldpoint_vc
        # worldpoint = np.dot(vehicle_Matrixinv, np.hstack([worldpoint_vc,1]))*np.array([self.scalex, self.scaley, 1])
        worldpoint = np.dot(worldpoint_n, vehicle_Matrixinv)*np.array([self.scalex, self.scaley, 1])
        
        imgpoint = np.dot(np.linalg.inv(self.BirdsEyeViewTransform), worldpoint.T)
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
    
    #out = (label_image==(1+np.argmax([i.area for i in region_props]))).astype(int)
    return lane_label_ret

@jit
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
    
    ego_right_index = right_dist.index(np.min(right_dist))
    ego_left_index = left_dist.index(np.min(left_dist))
    ego_right_lane = right_lane[ego_right_index]
    ego_left_lane = left_lane[ego_left_index]
    return ego_right_lane,ego_left_lane

def isgood(fit_param):
    a = fit_param[0]
    b = fit_param[1]
    isGood = abs(a) < 0.003 and abs(b) < 0.8
    return isGood

def get_fit_param(wordpoint_vc, init_fit_param, fit_model):
    # new_fit = np.array([])
    try:
        if wordpoint_vc.size == 0:
            fit_param = init_fit_param
            return init_fit_param
        else: 
            nonzero_x = wordpoint_vc[:,0]
            nonzero_y = wordpoint_vc[:,1]
            fit_model.fit(np.expand_dims(nonzero_x, axis=1), nonzero_y)
            fit_param = fit_model.estimator_.coeffs

            return fit_param
        
            # isGood = isgood(fit_param)
            # if isGood == True:
            #     return fit_param
            # else:
            #     return init_fit_param
    except:
        fit_param = init_fit_param
        return init_fit_param

    
def get_fit_param_bak(warpimage, init_fit_param, fit_model):
    
    label_image = label(warpimage)
    region_props = regionprops(label_image)
    new_fit = np.array([])
    # fit_model = RANSACRegressor(PolynomialRegression(degree=2), random_state=0)

    try:
        maxlabel = (label_image==(1+np.argmax([i.area for i in region_props]))).astype(int)
        
        fit_params = []
        if np.any(maxlabel):
            nonzero = np.nonzero(maxlabel!=0)
            nonzero_y = nonzero[0]
            nonzero_x = nonzero[1]

            # fit_param = np.polyfit(nonzero_y, nonzero_x, 2) # np.polyfit(nonzero_y, nonzero_x, 3)

            fit_model.fit(np.expand_dims(nonzero_y, axis=1), nonzero_x)
            fit_param = fit_model.estimator_.coeffs

            isGood = isgood(fit_param)
            
            new_y = nonzero[0]/42.3
            new_x = nonzero[1]/42.3-5
            new_fit = np.polyfit(new_y, new_x, 2)
            if isGood == True:
                return fit_param, new_fit
            else:
                return init_fit_param, new_fit
    except:
        fit_param = init_fit_param
        return init_fit_param, new_fit

def lanefit(binary_image,mtx,CameraPose, OutImageView, OutImageSize, fit_model):   
    warpimage, unwarp_matrix, birdseyeview = bev_perspective(binary_image.astype(np.uint8), mtx, CameraPose, OutImageView, OutImageSize)
    warpimage[warpimage[:,:]!=0]=255
    leftwarpimg = warpimage.copy()
    leftwarpimg[:,int(OutImageSize[1]/2):int(OutImageSize[1])]=0
    
    rightwarpimg = warpimage.copy()
    rightwarpimg[:,0:int(OutImageSize[1]/2)]=0
    
    # left_init_param = np.array([-4.10852052e-05, -7.97958759e-02,  1.29887915e+02])
    # right_init_param = np.array([-2.79490979e-05,  4.62164141e-02,  2.44594161e+02])
    left_init_param = np.array([])
    right_init_param = np.array([])
    ego_left_lane, left_new = get_fit_param(leftwarpimg, left_init_param, fit_model)
    ego_right_lane, right_new = get_fit_param(rightwarpimg, right_init_param, fit_model)
    
    ret = {
        'ego_right_lane': ego_right_lane,
        'ego_left_lane': ego_left_lane,
        'right_new': right_new,
        'left_new': left_new,}
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

def drawsinglelane(binary_image, warpimage, unwarp_matrix, ego_lane):   
    line_img = np.zeros_like(warpimage).astype(np.uint8)
    plot_y = np.linspace(0, warpimage.shape[0]-1, warpimage.shape[0])
    fit_param=ego_lane
    warp_xs = []
    warp_ys = []

    if not fit_param.any():
        lane_image = np.zeros_like(binary_image).astype(np.uint8)
        return lane_image
    else:    
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
            lane_image = cv2.circle(lane_image,points,1,255,-1)
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
        lane_image = cv2.circle(lane_image,points,1,255,-1)
    return lane_image
    
def imageadd(img1,img2):
    addimage = cv2.addWeighted(img1,1,img2,1,0)
    return addimage

def lanefit_bak(binary_image,mtx,CameraPose, OutImageView, OutImageSize):   
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

def get_points_xy(fit_param, xPoints, OutImageView, worldHW, birdseyeview, line_img, img):
    fit_x = fit_param[0] * xPoints ** 2 + fit_param[1] * xPoints ** 1 + fit_param[2]    
    idx_fitx = (fit_x>=-OutImageView.spaceToRightSide) & (fit_x<=OutImageView.spaceToLeftSide)
    
    X = xPoints[idx_fitx]
    y = fit_x[idx_fitx]

    worldpoints = np.ones((X.shape[0],3))
    worldpoints[:,0] = X
    worldpoints[:,1] = y

    T_matrix = np.array([[0, -1, worldHW[1]/2],
                        [-1, 0, worldHW[0]],
                        [0, 0, 1]])
    bevpoints = np.dot(worldpoints, T_matrix.T)
    imgpoints = np.int_(bevpoints[:,0:2] * np.array([birdseyeview.scalex, birdseyeview.scaley]))
    line_pts = (imgpoints[:,1], imgpoints[:,0])   
    line_img[line_pts] = 255
    mask_img = cv2.warpPerspective(line_img, birdseyeview.unwarp_matrix, (img.shape[1], img.shape[0]))
    
    src_xy = np.argwhere(mask_img != 0)
    src_xy = np.flip(src_xy[:,0:2],1)
    points_xy = [tuple(x) for x in src_xy]
    return points_xy

def insertLaneBoundary(img, warpimage, lane_param, OutImageView, birdseyeview, lanecolor=(0,0,255), shape_line = False):
    line_img = np.zeros_like(warpimage).astype(np.uint8)
    fit_param = lane_param
    xPoints = np.linspace(OutImageView.bottomOffset, OutImageView.distAheadOfSensor,100)[:, np.newaxis]
    
    worldHW = np.array([OutImageView.distAheadOfSensor, OutImageView.spaceToRightSide +OutImageView.spaceToLeftSide ])

    if not fit_param.any():
        # lane_image = np.zeros_like(img).astype(np.uint8)
        return img
    else:
        # fit_x = fit_param[0] * xPoints ** 2 + fit_param[1] * xPoints ** 1 + fit_param[2]    
        # idx_fitx = (fit_x>=-OutImageView.spaceToRightSide) & (fit_x<=OutImageView.spaceToLeftSide)
        
        # X = xPoints[idx_fitx]
        # y = fit_x[idx_fitx]

        # worldpoints = np.ones((X.shape[0],3))
        # worldpoints[:,0] = X
        # worldpoints[:,1] = y

        # T_matrix = np.array([[0, -1, worldHW[1]/2],
        #                     [-1, 0, worldHW[0]],
        #                     [0, 0, 1]])
        # bevpoints = np.dot(worldpoints, T_matrix.T)
        # imgpoints = np.int_(bevpoints[:,0:2] * np.array([birdseyeview.scalex, birdseyeview.scaley]))
        # line_pts = (imgpoints[:,1], imgpoints[:,0])   
        # line_img[line_pts] = 255
        # mask_img = cv2.warpPerspective(line_img, birdseyeview.unwarp_matrix, (img.shape[1], img.shape[0]))
        
        # src_xy = np.argwhere(mask_img != 0)
        # src_xy = np.flip(src_xy[:,0:2],1)
        # points_xy = [tuple(x) for x in src_xy]
        
        points_xy = get_points_xy(fit_param, xPoints, OutImageView, worldHW, birdseyeview, line_img, img)
        if not points_xy:
            return img
        else: 
            lane_image = np.zeros((img.shape[0],img.shape[1])).astype(np.uint8)

            if not shape_line:    
                for points in points_xy:
                    lane_image = cv2.circle(img,points,1,lanecolor,-1)
            else:
                for index, item in enumerate(points_xy): 
                    if index == len(points_xy) -1:
                        break
                    lane_image = cv2.line(img, item, points_xy[index + 1], lanecolor, 2) 
            return lane_image  

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
