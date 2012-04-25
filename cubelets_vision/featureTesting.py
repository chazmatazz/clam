#!/usr/bin/env python
import roslib; roslib.load_manifest('cubelets_vision')
#import roslib; roslib.load_manifest('beginner_tutorials')
import sys
import rospy
import cv,cv2
import numpy
import math
from scipy.spatial import KDTree
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from CubeletDetection import *

FONT = cv.InitFont(cv.CV_FONT_HERSHEY_PLAIN, 1, 1, 0, 3, 8)



def starFeatureDemo():
	# Test Images
	templatePath = BLACK_CUBE
	targetPath = "images/rotatedCube.png"

	templateImg = cv.LoadImageM(templatePath, cv.CV_LOAD_IMAGE_COLOR)
	targetImg = cv.LoadImageM(targetPath, cv.CV_LOAD_IMAGE_COLOR)

	storage = cv.CreateMemStorage()
	cv.ShowImage("Original",targetImg)
	kp = cv.GetStarKeypoints(targetImg, storage)
	if (len(kp) > 0):
		for (x,y),size,r in kp:
			print (x,y),size,r
			cv.Circle(targetImg, (x,y), size, (0,0,255))

	cv.ShowImage("Star Features",targetImg)
	cv.WaitKey()

def demoHoughCircles():
	#This does not work well.
	templatePath = "images/cornerCube.png"
	gray = cv.LoadImageM(templatePath, cv.CV_LOAD_IMAGE_GRAYSCALE)
	#cv.Canny(gray,gray,30,40)

	cv.SetZero(storage)
	cv.ShowImage('Circles',gray)
	cv.WaitKey()
	storage = cv.CreateMat(1, 10, cv.CV_32FC3)
	centers = cv.HoughCircles(gray, storage, cv.CV_HOUGH_GRADIENT, 2, 1, 10, 10, 0, 1000)
	print storage[0,0]
	print storage[0,1]
	print centers
	print storage.height,storage.width
#	print storage[0,1]
	for i in range(0,storage.width):
		print storage[0,i]
		cv.Circle(gray,(int(storage[0,i][0]),int(storage[0,i][1])),int(storage[0,0][2]),255)
	cv.ShowImage('Circles',gray)
	cv.WaitKey()
	return

def imageViewer(imagePath="images/rotatedCube.png", showSurf = True):
	img = cv.LoadImageM(imagePath, cv.CV_LOAD_IMAGE_GRAYSCALE)
	cv.ShowImage('Original', img)
	cv.MoveWindow("Original", 60, 0)

	cv.Smooth(img,img,cv.CV_GAUSSIAN,3,-1)
	cv.Canny(img,img,395,400)
	if showSurf:
		(keypoints, descriptors) = cv.ExtractSURF(img, None, cv.CreateMemStorage(), (0, 1000, 3, 1))
		for i in range(0,len(keypoints)):
			print keypoints[i][0]
			d1 = pixelDist(keypoints[i][0],(409, 155))
			d2 = pixelDist(keypoints[i][0],(455, 138))
			d3 = pixelDist(keypoints[i][0],(430, 192))
			d4 = pixelDist(keypoints[i][0],(479, 175))

			if (d1<10)|(d2<10)|(d3<10)|(d4<10):
				img = overlayKeyPoint(img,keypoints[i],(0,0,255))
			else:
				img = overlayKeyPoint(img,keypoints[i],(0,255,0))
		cv.ShowImage(imagePath, img)
	else:
		cv.ShowImage(imagePath, img)
	cv.MoveWindow(imagePath, 700, 0)
	cv.WaitKey()
def pixelDist(p1,p2):
	dx = p1[0] - p2[0]
	dy = p1[1] - p2[1]
	d = math.sqrt(dx*dx + dy*dy)
	return d

def overlayKeyPoint(imgMat,keyPoint,color,offset=(0,0)):
	# Overlay a surf feature marker onto an image
	# If grayscale, convert to color
	if isinstance(imgMat[0,0], float):
		overlaid = cv.CreateMat(imgMat.height, imgMat.width, cv.CV_8UC3)
		cv.CvtColor(imgMat, overlaid, cv.CV_GRAY2BGR)
	else:
		overlaid = imgMat
	
	((x, y), laplacian, size, fdir, hessian) = keyPoint
	r = int(1.0*size)
	px = int(x + offset[0])
	py = int(y + offset[1])
	cv.Circle(overlaid, (px, py), r, color)

	radians = fdir/180.0*math.pi
	rpx = int(px+r*math.cos(radians))
	rpy = int(py-r*math.sin(radians))

	cv.Line(overlaid, (px, py), (rpx, rpy), color)

	return overlaid

def overlayKeyPoints(imgMat,keyPoints,color,offset=(0,0)):
	# Overlay a set of surf feature marker onto an image
	# If grayscale, convert to color
	if isinstance(imgMat[0,0], float):
		imSize = cv.GetSize(imgMat)
		overlaid = cv.CreateMat(imSize[1], imSize[0], cv.CV_8UC3)
		cv.CvtColor(imgMat, overlaid, cv.CV_GRAY2BGR)
	else:
		overlaid = imgMat
	
	for ((x, y), laplacian, size, fdir, hessian) in keyPoints:
#		if size < 60:
#			if size > 10:
		r = int(1.0*size)
		px = int(x + offset[0])
		py = int(y + offset[1])
		cv.Circle(overlaid, (px, py), r, color)
		radians = fdir/180.0*math.pi
		rpx = int(px+r*math.cos(radians))
		rpy = int(py-r*math.sin(radians))
		cv.Line(overlaid, (px, py), (rpx, rpy), color)
	return overlaid

def templateMatchDemo():
	imgPath = "images/rotatedCube.png"
	templatePath = "images/nubTemplate"
	img = cv.LoadImageM(imgPath, cv.CV_LOAD_IMAGE_GRAYSCALE)

	cv.ShowImage("img",img)
	cv.MoveWindow("img", 60, 0)
	#cv.EqualizeHist( img, img )
	cv.ShowImage("after hist",img)
	cv.MoveWindow("after hist", 700, 0)

	combinedResult = cv.CreateMat(img.height, img.width, cv.CV_32FC1)
	cv.Set(combinedResult,1)

	for i in range(0,5):
		templatePath = "images/nubTemplate" + str(i+1) + ".png"
		template = cv.LoadImageM(templatePath, cv.CV_LOAD_IMAGE_GRAYSCALE)

		rh = 1+img.height-template.height
		rw = 1+img.width-template.width
		result = cv.CreateMat(rh, rw, cv.CV_32FC1)
		cv.MatchTemplate(img, template, result, cv.CV_TM_SQDIFF_NORMED)
		offsetX = combinedResult.width-rw
		offsetY = combinedResult.height-rh
		if (i == 0):

			for y in range(0,rh):
				for x in range(0,rw):
					combinedResult[y+offsetY,x+offsetX] = result[y,x]
		else:
			for y in range(0,rh):
				for x in range(0,rw):
					combinedResult[y+offsetY,x+offsetX] = combinedResult[y+offsetY,x+offsetX]*result[y,x]

	cv.ShowImage("result",combinedResult)
	cv.WaitKey()

	# Threshold
	for y in range(0,combinedResult.height):
		for x in range(0,combinedResult.width):
			if (combinedResult[y,x] > 0.4):
				combinedResult[y,x] = 1
	cv.ShowImage("result",combinedResult)
	cv.WaitKey()


if __name__ == '__main__':
	try:
		active = True
		while active:
			print "Select from the following demos:"
			print "s: Star Feature Demo"
			print "h: Hough Circles Demo"
			print "i: Surf Feature Image Viewer"
			print "t: TemplateMatch Demo"
			print "q: Quit"

			c = raw_input()

			if c.strip() == 's':
				starFeatureDemo()
			elif c.strip() == 'h':
				demoHoughCircles()
			elif c.strip() == 'i':
				imageViewer()
			elif c.strip() == 't':
				templateMatchDemo()
			elif c.strip() == 'q':
				active = False
			else:
				print "No valid option selected."

	except rospy.ROSInterruptException: 
		pass

