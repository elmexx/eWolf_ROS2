# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 11:42:18 2020

@author: gao
"""
import numpy as np
import cv2

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
  
    def birdseyeviewimage(self,image,IntrinsicMatrix,CameraPose,OutImgView,OutImgSize):
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
        
        rotation = np.linalg.multi_dot([rotY(180),rotZ(-90),rotZ(Yaw),rotX(90-Pitch),rotZ(Roll)])
        rotationMatrix = np.linalg.multi_dot([rotZ(Yaw),rotX(90-Pitch),rotZ(Roll)])
        sl = [0,0]
        translationInWorldUnits = [sl[1], sl[0], Height]
        translation = np.dot(translationInWorldUnits,rotationMatrix)
        camMatrix = np.dot(np.vstack([rotation,translation]),IntrinsicMatrix)
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
    
        

