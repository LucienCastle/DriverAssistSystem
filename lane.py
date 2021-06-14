import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from cam_cal import undistort_img
from copy import copy
from timeit import default_timer as timer
# from moviepy.editor import VideoFileClip

left_a,left_b,left_c=[],[],[]
right_a,right_b,right_c=[],[],[]

def pipeline(img,s_thresh=(100,255),sx_thresh=(15,255)):
    img=undistort_img(img,'calibration_pickle.p',False)
    img=np.copy(img)

    hls=cv2.cvtColor(img,cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel=hls[:,:,1]
    s_channel=hls[:,:,2]
    h_channel=hls[:,:,0]

    sobelx=cv2.Sobel(l_channel,cv2.CV_64F,1,1)
    abs_sobelx=np.absolute(sobelx)
    scaled_sobel=np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    sxbinary=np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel>=sx_thresh[0])&(scaled_sobel<=sx_thresh[1])]=1

    s_binary=np.zeros_like(s_channel)
    s_binary[(s_channel>=s_thresh[0])&(s_channel<=s_thresh[1])]=1

    color_binary=np.dstack((np.zeros_like(sxbinary),sxbinary,s_binary))*255

    combined_binary=np.zeros_like(sxbinary)
    combined_binary[(s_binary==1)|(sxbinary==1)]=1
    return combined_binary

def perspective_warp(img,dst_size=(1280,720),src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),dst=np.float32([(0,0),(1,0),(0,1),(1,1)])):
    img_size=np.float32([(img.shape[1],img.shape[0])])
    src=src*img_size
    dst=dst*np.float32(dst_size)

    M=cv2.getPerspectiveTransform(src,dst)
    warped=cv2.warpPerspective(img,M,dst_size)
    return warped

def inv_perspective_warp(img,dst_size=(1280,720),src=np.float32([(0,0),(1,0),(0,1),(1,1)]),dst=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])):
    img_size=np.float32([(img.shape[1],img.shape[0])])
    src=src*img_size
    dst=dst*np.float32(dst_size)

    M=cv2.getPerspectiveTransform(src,dst)
    warped=cv2.warpPerspective(img,M,dst_size)
    return warped

def get_hist(img):
    hist=np.sum(img[img.shape[0]//2:,:],axis=0)
    return hist

def sliding_window(img,nwindows=9,margin=150,minpix=1,draw_windows=True):
    global left_a,left_b,left_c,right_a,right_b,right_c
    left_fit_=np.empty(3)
    right_fit_=np.empty(3)
    out_img=np.dstack((img,img,img))*255

    histogram=get_hist(img)
    midpoint=int(histogram.shape[0]/2)
    leftx_base=np.argmax(histogram[:midpoint])
    rightx_base=np.argmax(histogram[midpoint:])+midpoint

    window_height=np.int(img.shape[0]/nwindows)

    nonzero=img.nonzero()
    nonzeroy=np.array(nonzero[0])
    nonzerox=np.array(nonzero[1])
    leftx_current=leftx_base
    rightx_current=rightx_base

    left_lane_ind=[]
    right_lane_ind=[]

    for window in range(nwindows):
        win_y_low=img.shape[0]-(window+1)*window_height
        win_y_high=img.shape[0]-window*window_height
        win_xleft_low=leftx_current-margin
        win_xleft_high=leftx_current+margin
        win_xright_low=rightx_current-margin
        win_xright_high=rightx_current+margin

        if draw_windows==True:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(100,255,255),3)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(100,255,255),3)

        good_left_ind=((nonzeroy>=win_y_low)&(nonzeroy<win_y_high)&(nonzerox>=win_xleft_low)&(nonzerox<win_xleft_high)).nonzero()[0]
        good_right_ind=((nonzeroy>=win_y_low)&(nonzeroy<win_y_high)&(nonzerox>=win_xright_low)&(nonzerox<win_xright_high)).nonzero()[0]

        left_lane_ind.append(good_left_ind)
        right_lane_ind.append(good_right_ind)

        if len(good_left_ind)>minpix:
            leftx_current=np.int(np.mean(nonzerox[good_left_ind]))

        if len(good_right_ind)>minpix:
            rightx_current=np.int(np.mean(nonzerox[good_right_ind]))

    left_lane_ind=np.concatenate(left_lane_ind)
    right_lane_ind=np.concatenate(right_lane_ind)

    # ploty=np.linspace(0,img.shape[0]-1,img.shape[0])
    #
    # if len(left_lane_ind)>0:
    #     leftx=nonzerox[left_lane_ind]
    #     lefty=nonzeroy[left_lane_ind]
    #
    #     left_fit=np.polyfit(lefty,leftx,2)
    #
    #     left_a.append(left_fit[0])
    #     left_b.append(left_fit[1])
    #     left_c.append(left_fit[2])
    #
    #     left_fit_[0]=np.mean(left_a[-10:])
    #     left_fit_[1]=np.mean(left_b[-10:])
    #     left_fit_[2]=np.mean(left_c[-10:])
    #
    #     left_fitx=left_fit_[0]*ploty**2+left_fit_[1]*ploty+left_fit_[2]
    #     out_img[nonzeroy[left_lane_ind],nonzerox[left_lane_ind]]=[255,0,100]
    # else:
    #     left_fitx=[]
    #     left_fit_=[]
    #
    # if len(right_lane_ind)>0:
    #     rightx=nonzerox[right_lane_ind]
    #     righty=nonzeroy[right_lane_ind]
    #
    #     right_fit=np.polyfit(righty,rightx,2)
    #
    #     right_a.append(right_fit[0])
    #     right_b.append(right_fit[1])
    #     right_c.append(right_fit[2])
    #
    #     right_fit_[0]=np.mean(right_a[-10:])
    #     right_fit_[1]=np.mean(right_b[-10:])
    #     right_fit_[2]=np.mean(right_c[-10:])
    #
    #     right_fitx=right_fit_[0]*ploty**2+right_fit_[1]*ploty+right_fit_[2]
    #     out_img[nonzeroy[right_lane_ind],nonzerox[right_lane_ind]]=[0,100,255]
    # else:
    #     right_fitx=[]
    #     right_fit_=[]


    leftx=nonzerox[left_lane_ind]
    lefty=nonzeroy[left_lane_ind]
    rightx=nonzerox[right_lane_ind]
    righty=nonzeroy[right_lane_ind]

    left_fit=np.polyfit(lefty,leftx,2)
    right_fit=np.polyfit(righty,rightx,2)

    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])

    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])

    left_fit_[0]=np.mean(left_a[-10:])
    left_fit_[1]=np.mean(left_b[-10:])
    left_fit_[2]=np.mean(left_c[-10:])

    right_fit_[0]=np.mean(right_a[-10:])
    right_fit_[1]=np.mean(right_b[-10:])
    right_fit_[2]=np.mean(right_c[-10:])

    ploty=np.linspace(0,img.shape[0]-1,img.shape[0])
    left_fitx=left_fit_[0]*ploty**2+left_fit_[1]*ploty+left_fit_[2]
    right_fitx=right_fit_[0]*ploty**2+right_fit_[1]*ploty+right_fit_[2]

    out_img[nonzeroy[left_lane_ind],nonzerox[left_lane_ind]]=[255,0,100]
    out_img[nonzeroy[right_lane_ind],nonzerox[right_lane_ind]]=[0,100,255]

    return out_img,(left_fitx,right_fitx),(left_fit_,right_fit_),ploty

def get_curve(img,leftx,rightx):
    ploty=np.linspace(0,img.shape[0]-1,img.shape[0])
    y_eval=np.max(ploty)
    ym_per_pix=30.5/720
    xm_per_pix=3.7/720

    # car_pos=img.shape[1]/2
    #
    # if len(leftx)>0:
    #     left_fit_cr=np.polyfit(ploty*ym_per_pix,leftx*xm_per_pix,2)
    #     left_curve_rad=((1+(2*left_fit_cr[0]*y_eval*ym_per_pix+left_fit_cr[1])**2)**1.5)/np.absolute(2*left_fit_cr[0])
    #     l_fit_x_int=left_fit_cr[0]*img.shape[0]**2+left_fit_cr[1]*img.shape[0]+left_fit_cr[2]
    # else:
    #     left_curve_rad=0
    #     l_fit_x_int=0
    #
    # if len(rightx)>0:
    #     right_fit_cr=np.polyfit(ploty*ym_per_pix,rightx*xm_per_pix,2)
    #     right_curve_rad=((1+(2*right_fit_cr[0]*y_eval*ym_per_pix+right_fit_cr[1])**2)**1.5)/np.absolute(2*right_fit_cr[0])
    #     r_fit_x_int=right_fit_cr[0]*img.shape[0]**2+right_fit_cr[1]*img.shape[0]+right_fit_cr[2]
    # else:
    #     right_curve_rad=0
    #     r_fit_x_int=0
    #
    # if leftx!=None and rightx!=None:
    #     lane_center_pos=(r_fit_x_int+l_fit_x_int)/2
    #     center=(car_pos-lane_center_pos)*xm_per_pix/10
    # else:
    #     center=0

    left_fit_cr=np.polyfit(ploty*ym_per_pix,leftx*xm_per_pix,2)
    right_fit_cr=np.polyfit(ploty*ym_per_pix,rightx*xm_per_pix,2)

    left_curve_rad=((1+(2*left_fit_cr[0]*y_eval*ym_per_pix+left_fit_cr[1])**2)**1.5)/np.absolute(2*left_fit_cr[0])
    right_curve_rad=((1+(2*right_fit_cr[0]*y_eval*ym_per_pix+right_fit_cr[1])**2)**1.5)/np.absolute(2*right_fit_cr[0])

    car_pos=img.shape[1]/2
    l_fit_x_int=left_fit_cr[0]*img.shape[0]**2+left_fit_cr[1]*img.shape[0]+left_fit_cr[2]
    r_fit_x_int=right_fit_cr[0]*img.shape[0]**2+right_fit_cr[1]*img.shape[0]+right_fit_cr[2]

    lane_center_pos=(r_fit_x_int+l_fit_x_int)/2
    center=(car_pos-lane_center_pos)*xm_per_pix/10

    return (left_curve_rad,right_curve_rad,center)

def draw_lanes(img,left_fit,right_fit,Offset):
    ploty=np.linspace(0,img.shape[0]-1,img.shape[0])
    color_img=np.zeros_like(img)

    # if len(left_fit)>0:
    #     left=np.array([np.transpose(np.vstack([left_fit,ploty]))])
    # else:
    #     left=np.zeros_like(np.vstack([right_fit,ploty]))
    # if len(right_fit)>0:
    #     right=np.array([np.flipud(np.transpose(np.vstack([right_fit,ploty])))])
    # else:
    #     right=np.zeros_like(np.vstack([left_fit,ploty]))
    #
    # points=np,hstack((left,right))
    # # if left_fit!=None and right_fit!=None:
    # if np.absolute(Offset)>0.1:
    #     cv2.fillPoly(color_img,np.int_(points),(0,0,255))
    # else:
    #     cv2.fillPoly(color_img,np.int_(points),(0,255,0))

    left=np.array([np.transpose(np.vstack([left_fit,ploty]))])
    right=np.array([np.flipud(np.transpose(np.vstack([right_fit,ploty])))])

    points=np.hstack((left,right))
    if np.absolute(Offset)>0.4:
        cv2.fillPoly(color_img,np.int_(points),(0,0,255))
    else:
        cv2.fillPoly(color_img,np.int_(points),(0,255,0))
    inv_perspective=inv_perspective_warp(color_img)
    inv_perspective=cv2.addWeighted(img,1,inv_perspective,0.7,0)

    return inv_perspective

def vid_pipeline(img):
    global running_avg
    global index
    img_=pipeline(img)
    img_=perspective_warp(img_)
    out_img,curves,lanes,ploty=sliding_window(img_,draw_windows=False)

    curve_rad=get_curve(img,curves[0],curves[1])
    lane_curve=np.mean([curve_rad[0],curve_rad[1]])
    img=draw_lanes(img,curves[0],curves[1],curve_rad[2])

    font=cv2.FONT_HERSHEY_SIMPLEX
    fontColor=(0,0,0)
    fontSize=0.5
    cv2.putText(img,'Lane Curvature:{:.0f} m'.format(lane_curve),(570,620),font,fontSize,fontColor,2)
    cv2.putText(img,'Vehicle Offset:{:.4f} m'.format(curve_rad[2]),(570,650),font,fontSize,fontColor,2)

    return img

# if __name__=="__main__":
    # img = cv2.imread('testing/test2.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # dst = pipeline(img)
    # plt.imshow(dst)
    # dst = perspective_warp(dst, dst_size=(1280,720))
    # print('dds')
    # Visualize undistortion
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    # ax1.imshow(img)
    # ax1.set_title('Original Image', fontsize=30)
    # ax2.imshow(dst, cmap='gray')
    # ax2.set_title('Warped Image', fontsize=30)
    # plt.show()
    #
    # out_img,curves,lanes,ploty=sliding_window(dst)
    # print(np.asarray(curves).shape)
    # curverad=get_curve(img, curves[0],curves[1])
    # print(curverad)
    # img_ = draw_lanes(img, curves[0], curves[1])
    # plt.imshow(img_, cmap='hsv')
    # plt.show()
    # n=10
    # cap=cv2.VideoCapture("challenge_video.mp4")
    # while(cap.isOpened()):
    #     ret,frame=cap.read()
    #     if ret==True:
    #         n-=1
    #         if n==0:
    #             n=5
    #             img=vid_pipeline(frame)
    #             cv2.imshow('Final',img)
    #         # else:
    #         #     cv2.imshow('Final',frame)
    #         if cv2.waitKey(1)==ord('q'):
    #             break
    #     else:
    #         break
    # cap.release()
    # cv2.destroyAllWindows()
