import numpy as np
import pandas as pd
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from os import path

def calib_cam(nx,ny,basepath):
    obj_pts=np.zeros((nx*ny,3),np.float32)
    obj_pts[:,:2]=np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    objpts=[]
    imgpts=[]

    images=glob.glob(path.join(basepath,'calibration*.jpg'))

    for idx,fname in enumerate(images):
        img=cv2.imread(fname)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        ret,corners=cv2.findChessboardCorners(gray,(nx,ny),None)

        if ret==True:
            objpts.append(obj_pts)
            imgpts.append(corners)

            img=cv2.drawChessboardCorners(img,(nx,ny),corners,ret)
            cv2.imshow('input image',img)
            cv2.waitKey(500)
    cv2.destroyAllWindows

    img_size=(img.shape[1],img.shape[0])
    ret,mtx,dist,rvecs,tvecs=cv2.calibrateCamera(objpts,imgpts,img_size,None,None)
    dist_pickle={}
    dist_pickle["mtx"]=mtx
    dist_pickle["dist"]=dist
    destination=path.join(basepath,'calibration_pickle.p')
    pickle.dump(dist_pickle,open(destination,"wb"))
    return mtx,dist

def load_calib(calib_file):

    with open(calib_file,'rb')as file:
        data=pickle.load(file)
        mtx=data['mtx']
        dist=data['dist']
    return mtx,dist

def undistort_img(img,calib_file,visualization_flag):
    mtx,dist=load_calib(calib_file)
    # img=cv2.imread(imagepath)
    img_undist=cv2.undistort(img,mtx,dist,None,mtx)
    img_undistRGB=cv2.cvtColor(img_undist,cv2.COLOR_BGR2RGB)

    if visualization_flag:
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        f,(ax1,ax2)=plt.subplots(1,2)
        ax1.imshow(imgRGB)
        ax1.set_title('Original IMage',fontsize=30)
        ax1.axis('off')
        ax2.imshow(img_undistRGB)
        ax2.set_title('Undistorted IMage',fontsize=30)
        ax2.axis('off')
        plt.show()

    return img_undistRGB

if __name__=="__main__":
    nx,ny=9,6
    basepath='camera_cal/'
    calib_cam(nx,ny,basepath)
