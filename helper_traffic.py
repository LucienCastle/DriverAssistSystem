import cv2
cap=cv2.VideoCapture("challenge_video.mp4")
savePath = 'inter/'
if cap.isOpened():
	n = 0
	i = 0
	while cap.isOpened():
		n = n + 1
		ret,frame=cap.read()
		if ret == True:
			i = i + 1
			cv2.imwrite(savePath+str(i)+'.jpg', frame)
			if cv2.waitKey(1)==ord('q'):
				break
		elif ret==False:
			break
	cap.release()
cv2.destroyAllWindows()