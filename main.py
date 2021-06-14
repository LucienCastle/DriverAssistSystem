import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

import lane
from lane import vid_pipeline

import time

import pandas as pd
from packages import object_config as config
from packages.Object_detection import detect_people
from scipy.spatial import distance as dist
import imutils
import os
import io

import math

from IPython.display import Image, display
from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

import traffic
from traffic import YoloV3
from traffic import transform_images
from traffic import draw_outputs


# load classes for object detection
labelsPath =os.path.sep.join([config.MODEL_PATH,"coco.names"])
LABELS =open(labelsPath).read().strip().split("\n")

# load weights for object detection
weightsPath =os.path.sep.join([config.MODEL_PATH,"yolov2-tiny.weights"])

# load config file for object detection
configPath= os.path.sep.join([config.MODEL_PATH,"tiny.cfg"])
print("[INFO] loading YOLO from disk")

# init dnn using loaded weights
net =cv2.dnn.readNetFromDarknet(configPath,weightsPath)


if config.USE_GPU:
	# checks if GPU is available for usage
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# get layers
ln = net.getLayerNames()

ln = [ln[i[0]-1]for i in net.getUnconnectedOutLayers()]

# filter relevant objects for detection
objects_list = []
for i in [2,5,7,9,11]:
    objects_list.append(LABELS[i])

def euclidean(x,y,dst_size=(700,400),src=np.float32([(0,0.5),(1,0.5),(0,1),(1,1)]),dst=np.float32([(0,0),(1,0),(0,1),(1,1)])):
	'''
	calculates the euclidean distance between ego vehicle and the object using perspective transform

	:params:
	 x,y      : coordinates of vehicle centroid
	 dst_size : destination values for perspective transform
	 src      : source points for transform
	 dst      : destination points for transform

	:returns:
	 distance between ego vehicle and detected object/vehicle
	'''

	# claculate source points
	img_size = np.float32([(700,400)])
	src = src*img_size

	# calculate destination points
	dst = dst*np.float32(dst_size)

	# get transformation matrix
	matrix = cv2.getPerspectiveTransform(src,dst)

	# transform points
	p = np.array([x,y], np.float32)
	p = p.reshape(-1,1,2).astype(np.float32)
	transformed_points_obj = cv2.perspectiveTransform(p, matrix)
	pts = transformed_points_obj[0][0]
	
	# define meters per pixel in X and Y direction
	ym_per_pix = 30.7/480
	xm_per_pix = 3.6/640

	return math.sqrt(pow(pts[0]*xm_per_pix-700*xm_per_pix,2)+pow(pts[1]*ym_per_pix-400*ym_per_pix,2))

def predict_objects(img):
	'''
	predicts the bounding box, class and location of object

	:params:
	 img : image frame

	:returns:
	 frame : image with detected object
	'''
	frame = imutils.resize(img,width =700, height=400)
	labels=[]
	labels.append(objects_list)

	for label in objects_list:
		results = detect_people(frame,net,ln,personIdx=LABELS.index(label))

		for(i,(prob,bbox,centroid)) in enumerate(results):
			(startX,startY,endX,endY)=bbox
			(cX,cY)=centroid
			print(cX,cY)
			
			if ( cY > 250) or ( cX < 360 and cX > 350):
				color = (0,0,255)
			else:
				color=(0,255,0)
			# calculate distance between ego vehicle and object
			dist = euclidean(cX,cY)
			# draw the predictions
			cv2.rectangle(frame,(startX,startY),(endX,endY),color,2)
			font = cv2.FONT_HERSHEY_SIMPLEX
			fontSize = 0.5
			fontColor = (0,0,0)
			cv2.putText(frame,'Dist:{:.4f} m '.format(dist),(startX,startY),font,fontSize,fontColor,2)
	return frame

def Convertltd(lst): 
	res_dct = {i: lst[i] for i in range(0, len(lst))} 
	return res_dct

def traffic_pipeline(img_raw):
	'''
	pipeline function for traffic sign detection and recognition by performing transformations

	:params:
	 i : index of frame

	:returns:
	 predictions
	'''

	size = 416
	# img_raw = tf.image.decode_image(open(interPath+str(i)+'.jpg', 'rb').read(), channels=3)
	img = tf.expand_dims(img_raw, 0)
	img = transform_images(img, size)
	boxes, scores, classes1, nums = modell(img)
	img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
	return (boxes, scores, classes1, nums)

   
if __name__ == "__main__":
	# load classes for traffic sign detection and recognition
	classes1 = 'obj.names'
	class_names = [c.strip() for c in open(classes1).readlines()]

	# load weights for traffic sign detection and recognition
	with open ('lr001_2_1_5', 'rb') as fp:
		weights = pickle.load(fp)

	# initialize model with loaded weights
	modell = YoloV3(training=False)
	modell.set_weights(weights)

	start=time.time()
	output = 'test_v9.avi'
	
	# open video file to be tested
	cap=cv2.VideoCapture("challenge_video.mp4")
	interPath = 'inter/'
	
	if cap.isOpened():
		w=int(cap.get(3))
		h=int(cap.get(4))
		size=(w,h)
		video=cv2.VideoWriter(output,cv2.VideoWriter_fourcc(*'XVID'),20.0,(640,480))
		n = 0
		i = 0
		while cap.isOpened():
			n = n + 1
			ret,frame=cap.read()
			if ret == True:
				i = i + 1
				cv2.imwrite(interPath+'interImg.jpg', frame)

				# call pipeline for lane detection
				img = vid_pipeline(frame)

				# detect objects 
				img = predict_objects(img)
				
				img_raw = tf.image.decode_image(open(interPath+str(i)+'.jpg', 'rb').read(), channels=3)
				outputs = traffic_pipeline(img_raw)
				img = draw_outputs(img, outputs, class_names)

				img = cv2.resize(img, (640, 480))
				video.write(img)
				cv2.imshow('Result',img)
				if cv2.waitKey(1)==ord('q'):
					break
			elif ret==False:
				break
		cap.release()
	cv2.destroyAllWindows()
	end  = time.time()
print(round(end-start, 2), 'Seconds to finish')


