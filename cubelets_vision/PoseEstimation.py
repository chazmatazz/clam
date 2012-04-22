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
from CubeletsSURF import *

FONT = cv.InitFont(cv.CV_FONT_HERSHEY_PLAIN, 1, 1, 0, 3, 8)

def cameraCalibration():
	# For use with a checkerboard pattern, like the one at
	# http://read.pudn.com/downloads101/sourcecode/graph/texture_mapping/414115/TOOLBOX_calib/calibration_pattern/pattern_p1__.jpg
	# Dim specifies the number of internal squares points per side
	dims=(8,6)
	imSize = (640,480)

	# Number of points in chessboard
	num_pts=dims[0] * dims[1]

	# Assumes a calibration image set of 10 images under images/cameraCalibration
	calibrationSet = []
	for i in range(0,10):
		path = "images/cameraCalibration/" + str(i) + ".png"
		calibrationSet.append(path)
	print calibrationSet

	# Number of calibration patterns used
	nimage = 0
	nMax = len(calibrationSet)

	# Points per view
	npts = cv.CreateMat(nMax, 1, cv.CV_32SC1)
	for i in range(0,nMax):
		npts[i,0] = num_pts

	# 
	opts = cv.CreateMat(nMax * num_pts, 3, cv.CV_32FC1)
	cv.SetZero(opts)
	ipts = cv.CreateMat(nMax * num_pts, 2, cv.CV_32FC1)
	#
	rvecs = cv.CreateMat(nMax, 3, cv.CV_32FC1)
	tvecs = cv.CreateMat(nMax, 3, cv.CV_32FC1)

	gray = cv.CreateMat(imSize[1], imSize[0], cv.CV_8UC1)

	for i in range(0,len(calibrationSet)):
		img = cv.LoadImage(calibrationSet[i],cv.CV_LOAD_IMAGE_COLOR)
		cv.CvtColor(img,gray,cv.CV_BGR2GRAY)
		found,points=cv.FindChessboardCorners(gray,dims,cv.CV_CALIB_CB_ADAPTIVE_THRESH)
		if found!=0:
			cv.DrawChessboardCorners(img,dims,points,found)
			cv.ShowImage("Chessboard",img)
			cv.WaitKey()
			for i in range(0,num_pts):
				ipts[nimage*num_pts+i,0] = points[i][0]
				ipts[nimage*num_pts+i,1] = points[i][1]
			i = 0
			for s in range(0,min(dims)):
				for l in range(0,max(dims)):
					opts[nimage*num_pts+i,0] = l
					opts[nimage*num_pts+i,1] = s
					i = i + 1

			nimage = nimage + 1
	intrinsics = cv.CreateMat(3, 3, cv.CV_64FC1)
	distortion = cv.CreateMat(4, 1, cv.CV_64FC1)

	cv.SetZero(intrinsics)
	cv.SetZero(distortion)

	# focal lengths have 1/1 ratio
	intrinsics[0,0] = 1.0
	intrinsics[1,1] = 1.0

	cv.CalibrateCamera2(opts, ipts, npts, imSize,intrinsics, distortion,rvecs,tvecs,flags = 0)

	print ""
	print "Camera Matrix:"
	print intrinsics[0,0],intrinsics[0,1],intrinsics[0,2]
	print intrinsics[1,0],intrinsics[1,1],intrinsics[1,2]
	print intrinsics[2,0],intrinsics[2,1],intrinsics[2,2]
	print ""
	print "Distortion Matrix:"
	print distortion[0,0]
	print distortion[1,0]
	print distortion[2,0]
	print distortion[3,0]
	print ""

def interactiveCameraCalibration():
	# For use with a checkerboard pattern, like the one at
	# http://read.pudn.com/downloads101/sourcecode/graph/texture_mapping/414115/TOOLBOX_calib/calibration_pattern/pattern_p1__.jpg
	# Dim specifies the number of internal squares points per side
	dims=(8,6)
	capture=cv.CaptureFromCAM(0)
	frame = cv.QueryFrame(capture)
	gray = cv.CreateMat(frame.height, frame.width, cv.CV_8UC1)

	#Number of calibration patterns used
	nimage=0
	nMax=10
	#Number of points in chessboard
	num_pts=dims[0] * dims[1]

	opts = cv.CreateMat(nMax * num_pts, 3, cv.CV_32FC1)
	cv.SetZero(opts)
	ipts = cv.CreateMat(nMax * num_pts, 2, cv.CV_32FC1)
	npts = cv.CreateMat(nMax, 1, cv.CV_32SC1)

	for i in range(0,nMax):
		npts[i,0] = num_pts

	rvecs = cv.CreateMat(nMax, 3, cv.CV_32FC1)
	tvecs = cv.CreateMat(nMax, 3, cv.CV_32FC1)

	while (nimage < nMax):
		frame = cv.QueryFrame(capture)
		cv.CvtColor(frame,gray,cv.CV_BGR2GRAY)
		found,points=cv.FindChessboardCorners(gray,dims,cv.CV_CALIB_CB_ADAPTIVE_THRESH)

		if found!=0:
			cv.DrawChessboardCorners(frame,dims,points,found)
			cv.ShowImage("Camera",frame)
			print "Save This data set? (y,n)"
			c = raw_input()
			if c.strip() == 'y':
				print "Adding current frame to calibration data set:"
				print points[5]
				print ipts[0]
				for i in range(0,num_pts):
					ipts[nimage*num_pts+i,0] = points[i][0]
					ipts[nimage*num_pts+i,1] = points[i][1]
					print points
				i = 0
				for s in range(0,min(dims)):
					for l in range(0,max(dims)):
						opts[nimage*num_pts+i,0] = l
						opts[nimage*num_pts+i,1] = s
						i = i + 1

				nimage = nimage + 1
				print "Calibration example " + str(nimage) + "/10 added."
			elif c.strip() == 'n':
				print "Nevermind."
			else:
				print "No valid option selected. Assuming no."
			cv.WaitKey(2)

		cv.ShowImage("Camera",frame)
		cv.WaitKey(2)

	intrinsics = cv.CreateMat(3, 3, cv.CV_64FC1)
	distortion = cv.CreateMat(4, 1, cv.CV_64FC1)

	cv.SetZero(intrinsics)
	cv.SetZero(distortion)

	# focal lengths have 1/1 ratio
	intrinsics[0,0] = 1.0
	intrinsics[1,1] = 1.0

	cv.CalibrateCamera2(opts, ipts, npts, (640,480),intrinsics, distortion,rvecs,tvecs,flags = 0)
	
	print "Camera Matrix:"
	print intrinsics[0,0],intrinsics[0,1],intrinsics[0,2]
	print intrinsics[1,0],intrinsics[1,1],intrinsics[1,2]
	print intrinsics[2,0],intrinsics[2,1],intrinsics[2,2]

	print "Distortion Matrix:"
	print distortion[0,0]
	print distortion[1,0]
	print distortion[2,0]
	print distortion[3,0]

	# Undistort Camera Feed
	# The training chessboard should look completely regular
	mapx = cv.CreateImage((640,480), cv.IPL_DEPTH_32F, 1)
	mapy = cv.CreateImage((640,480), cv.IPL_DEPTH_32F, 1)
	cv.InitUndistortMap(intrinsics, distortion, mapx, mapy)
	while True:
		frame = cv.QueryFrame(capture)
		r = cv.CloneImage(frame)
		cv.Remap(frame, r, mapx, mapy)
		cv.ShowImage("Camera",r)
		cv.WaitKey(1)

def keyPts2CoordMat(keyPtsArray):
	numPts = len(keyPtsArray)
	pointMat = cv.CreateMat(2,numPts,cv.CV_32FC1)
	for i in range(0,numPts):
		(x,y) = keyPtsArray[i][0]
		print x,y
		pointMat[0,i] = x
		pointMat[1,i] = x
	return pointMat

def findHomography(sourcePts,targetPts):
    # source, target should be 2x4+ CV_32FC1 matrices,
	# specifying the 2D coordinates of the points in each image
    homography = cv.CreateMat(3,3,cv.CV_32FC1)
    cv.FindHomography(sourcePts,targetPts,homography)
    return homography

def getSquareCorners(cornerCoords,sourcePts,targetPts):
    # source, target should be 2x4+ CV_32FC1 matrices, specifying the 
	# corresponding2D coordinates of the points in the template and target image
	homography = findHomography(sourcePts,targetPts)

	# Transform points (default of corners) in example image into perspective of target image
	transformedCoords = numpy.zeros((4,2),dtype=numpy.int32)

	for i in range(0,4):
		transformedPt = numpy.dot(homography,cornerCoords[i,:])		
		transformedCoords[i,0] = int(transformedPt[0]/transformedPt[2])
		transformedCoords[i,1] = int(transformedPt[1]/transformedPt[2])

	# Returns the square's corners post transformation
	return transformedCoords, homography

def demoSquareCorners():
	# Test Images
	templatePath = "images/dottedSquare.png"
	targetPath = "images/dottedSquareSkewed.png"
	templateImg = cv.LoadImageM(templatePath, cv.CV_LOAD_IMAGE_COLOR)
	targetImg = cv.LoadImageM(targetPath, cv.CV_LOAD_IMAGE_COLOR)

	# Four of the internal points on the test images
	templatePts = cv.CreateMat(2,4,cv.CV_32FC1)
	targetPts = cv.CreateMat(2,4,cv.CV_32FC1)

	templatePts[0,0] = 98.0
	templatePts[1,0] = 92.0

	templatePts[0,1] = 309.0
	templatePts[1,1] = 118.0

	templatePts[0,2] = 324.0
	templatePts[1,2] = 250.0

	templatePts[0,3] = 121.0
	templatePts[1,3] = 327.0
#
	targetPts[0,0] = 157.0
	targetPts[1,0] = 128.0

	targetPts[0,1] = 317.0
	targetPts[1,1] = 98.0

	targetPts[0,2] = 311.0
	targetPts[1,2] = 192.0

	targetPts[0,3] = 126.0
	targetPts[1,3] = 305.0

	c1 = numpy.array([17,14,1])
	c2 = numpy.array([384,14,1])
	c3 = numpy.array([17,384,1])
	c4 = numpy.array([384,384,1])
	cornerCoords = numpy.array([c1,c2,c3,c4])
	corners, homography = getSquareCorners(cornerCoords,templatePts,targetPts)

	# View the Images
	for i in range(0,4):
		cv.Circle(targetImg, (corners[i,0],corners[i,1]), 10, (255,0,0))

	cv.ShowImage("Template Image", templateImg)
	cv.ShowImage("Target Image with Corners marked", targetImg)
	cv.WaitKey()

	rotationMatrix, translationVector = findPlanarPose(homography)
	print rotationMatrix[0,0],rotationMatrix[0,1],rotationMatrix[0,2]
	print rotationMatrix[1,0],rotationMatrix[1,1],rotationMatrix[1,2]
	print rotationMatrix[2,0],rotationMatrix[2,1],rotationMatrix[2,2]

	return

def demoCubeSideCorners():
	# Test Images
	templatePath = BLACK_CUBE
	targetPath = "images/rotatedCube.png"

	templateImg = cv.LoadImageM(templatePath, cv.CV_LOAD_IMAGE_COLOR)
	targetImg = cv.LoadImageM(targetPath, cv.CV_LOAD_IMAGE_COLOR)

	# Four of the internal points on the test images
	templatePts = cv.CreateMat(2,4,cv.CV_32FC1)
	targetPts = cv.CreateMat(2,4,cv.CV_32FC1)

	templatePts[0,0] = 85.0
	templatePts[1,0] = 48.0

	templatePts[0,1] = 167.0
	templatePts[1,1] = 81.0

	templatePts[0,2] = 132.0
	templatePts[1,2] = 165.0

	templatePts[0,3] = 49.0
	templatePts[1,3] = 130.0
#
	targetPts[0,0] = 432.0
	targetPts[1,0] = 192.0

	targetPts[0,1] = 409.0
	targetPts[1,1] = 155.0

	targetPts[0,2] = 455.0
	targetPts[1,2] = 137.0

	targetPts[0,3] = 480.0
	targetPts[1,3] = 175.0

	# Coordinates of the Cube Corners in the template image
	c1 = numpy.array([0,0,1])
	c2 = numpy.array([218,0,1])
	c3 = numpy.array([0,218,1])
	c4 = numpy.array([218,218,1])
	cornerCoords = numpy.array([c1,c2,c3,c4])

	# Find the Corners relating to the target points in the template image
	corners, homography = getSquareCorners(cornerCoords,templatePts,targetPts)

	# Mark The Corresponding Internal Points for Calculating Homography
	for i in range(0,4):
		templateCoord = (int(templatePts[0,i]),int(templatePts[1,i]))
		cv.Circle(templateImg, templateCoord, 2, (0,0,255),-1)
		cv.PutText(templateImg, str(i), templateCoord, FONT, (0,255,255))

		targetCoord = (int(targetPts[0,i]),int(targetPts[1,i]))
		cv.Circle(targetImg, targetCoord, 2, (0,0,255),-1)
		cv.PutText(targetImg, str(i), targetCoord, FONT, (0,255,255))

	# View the Images
	for i in range(0,4):
		cv.Circle(targetImg, (corners[i,0],corners[i,1]), 10, (255,0,0))

	cv.ShowImage("Template Image", templateImg)
	cv.MoveWindow("Template Image", 60, 0)
	cv.ShowImage("Target Image with Corners marked", targetImg)
	cv.MoveWindow("Target Image with Corners marked", 640+60, 0)
	cv.WaitKey()

	# Close Windows from previous demos
	#cv.DestroyAllWindows()
	return


def findObjPose(focal_length,imgPts,objPts = [(0,0,0),(1,0,0),(0,1,0),(0,0,1)]):
	# objPts: List of 32f triplets, coordinates of points on object in object space
	# imgPts: List of 32f pairs, object points projections on the 2D image plane
	positObject = cv.CreatePOSITObject(objPts)

	rvecs, tvecs = cv.POSIT(positObject,imgPts,focal_length,(cv.CV_TERMCRIT_ITER, 10, 0))
	#print "rvecs: ",rvecs
	#print "tvecs: ",tvecs

	# rvecs and tvecs are not returned as cv matrices, 
	# need to convert to matrix format
	poseMat = cv.CreateMat(3,4,cv.CV_32FC1)
	for r in range(0,3):
		for c in range(0,3):
			poseMat[r,c] = rvecs[r][c]
	
	for r in range(0,3):
		poseMat[r,3] = tvecs[r]

	# Memory Management
	#cv.ReleasePositObject(positObject)
	# Function missing in python bindings

	return poseMat

def projectPoints(objModelPoints,cameraMatrix,poseMat):
	# objModelPoints is a list of coordinates in the object space
	imageCoords = []
	homogenousCoords = cv.CreateMat(4,1,cv.CV_32FC1)
	cameraCoords = cv.CreateMat(3,1,cv.CV_32FC1)

	for i in range(0,len(objModelPoints)):
		for j in range(0,3):
			homogenousCoords[j,0] = objModelPoints[i][j]
		homogenousCoords[3,0] = 1

		cv.GEMM(poseMat, homogenousCoords, 1.0, None, 0.0, cameraCoords)

		#print "cameraCoords: ", cameraCoords[0,0], cameraCoords[1,0], cameraCoords[2,0]
		x = cameraMatrix[0,0]*(cameraCoords[0,0]/cameraCoords[2,0])
		y = cameraMatrix[1,1]*(cameraCoords[1,0]/cameraCoords[2,0])
		imageCoords.append((int(x),int(y)))

	# return list of coordinates projected into camera image plane
	return imageCoords

def positDemo():
	imgPath = 'images/cornerCube.png'
	img = cv.LoadImageM(imgPath, cv.CV_LOAD_IMAGE_COLOR)

	cameraMatrix = cv.CreateMat(3,3,cv.CV_32FC1)
	cv.SetZero(cameraMatrix)
	cameraMatrix[0,0] = 560.0
	cameraMatrix[0,2] = 300.0
	cameraMatrix[1,1] = 560.0
	cameraMatrix[1,2] = 230.0
	cameraMatrix[2,2] = 1.0

	focal_length = 560.0
	
#	objPts = cv.CreateMat(4,3,cv.CV_32FC1)
#	cv.SetZero(objPts)
	objPts = [(0,0,0),(1,0,0),(0,1,0),(0,0,1)]	# Points used for pose estimation
	xtrPts = [(1,1,0),(1,0,1),(0,1,1),(1,1,1)]	# Other points on the cube, for display

	imgPts = cv.CreateMat(4,2,cv.CV_32FC1)
	cv.SetZero(imgPts)
	imgPts = [(437,228),(338,236),(470,106),(518,280)]
	poseMat = findObjPose(focal_length,imgPts,objPts)

	# Verify Estimation
	for i in range(0,4):
		objPts.append(xtrPts[i])
	imageCoords = projectPoints(objPts,cameraMatrix,poseMat)
	print "Object Points (in Object Space):"
	print objPts
	print "Projections of Object Points"
	print imageCoords

	# Mark Cube Corners
	cv.Circle(img, imageCoords[0], 10, (255,255,0))	# Reference Point
	cv.Circle(img, imageCoords[1], 10, (0,0,255))	# +x
	cv.Circle(img, imageCoords[2], 10, (0,255,0))	# +y
	cv.Circle(img, imageCoords[3], 10, (255,0,0))	# +z
	cv.Circle(img, imageCoords[4], 10, (0,255,255))
	cv.Circle(img, imageCoords[5], 10, (0,255,255))
	cv.Circle(img, imageCoords[6], 10, (0,255,255))
#	cv.Circle(img, imageXY[7], 10, (0,255,255))

	cv.Line(img, imageCoords[0], imageCoords[1], (255,255,255))
	cv.Line(img, imageCoords[0], imageCoords[2], (255,255,255))
	cv.Line(img, imageCoords[0], imageCoords[3], (255,255,255))
	cv.Line(img, imageCoords[4], imageCoords[1], (255,255,255))
	cv.Line(img, imageCoords[4], imageCoords[2], (255,255,255))
	cv.Line(img, imageCoords[5], imageCoords[1], (255,255,255))
	cv.Line(img, imageCoords[5], imageCoords[3], (255,255,255))
	cv.Line(img, imageCoords[6], imageCoords[2], (255,255,255))
	cv.Line(img, imageCoords[6], imageCoords[3], (255,255,255))

	# Mark Original Points for Pose Calculation
	for i in range(0,4):
		cv.Circle(img, imgPts[i], 2, (255,0,255),-1)
		
	cv.ShowImage("Corners",img)
	cv.MoveWindow("Corners", 60, 0)
	cv.WaitKey()

	return

def demoPoseEstimationChain():
	cameraMatrix = cv.CreateMat(3,3,cv.CV_32FC1)
	cv.SetZero(cameraMatrix)
	cameraMatrix[0,0] = 560.0
	cameraMatrix[0,2] = 300.0
	cameraMatrix[1,1] = 560.0
	cameraMatrix[1,2] = 230.0
	cameraMatrix[2,2] = 1.0

	# Test Images
	templatePath = BLACK_CUBE
	targetPath = "images/rotatedCube.png"

	templateImg = cv.LoadImageM(templatePath, cv.CV_LOAD_IMAGE_COLOR)
	targetImg = cv.LoadImageM(targetPath, cv.CV_LOAD_IMAGE_COLOR)

	# Four of the internal points on the test images
	templatePts = cv.CreateMat(2,4,cv.CV_32FC1)
	targetPts1 = cv.CreateMat(2,4,cv.CV_32FC1)
	targetPts2 = cv.CreateMat(2,4,cv.CV_32FC1)

	templatePts[0,0] = 85.0;	templatePts[1,0] = 48.0
	templatePts[0,1] = 167.0;	templatePts[1,1] = 81.0
	templatePts[0,2] = 132.0;	templatePts[1,2] = 165.0
	templatePts[0,3] = 49.0;	templatePts[1,3] = 130.0
#
	targetPts1[0,0] = 432.0;	targetPts1[1,0] = 192.0
	targetPts1[0,1] = 409.0;	targetPts1[1,1] = 155.0
	targetPts1[0,2] = 455.0;	targetPts1[1,2] = 137.0
	targetPts1[0,3] = 480.0;	targetPts1[1,3] = 175.0
#
	targetPts2[0,0] = 367.0;	targetPts2[1,0] = 231.0
	targetPts2[0,1] = 382.0;	targetPts2[1,1] = 218.0
	targetPts2[0,2] = 415.0;	targetPts2[1,2] = 257.0
	targetPts2[0,3] = 397.0;	targetPts2[1,3] = 265.0

	# Have lists of template/target points, corresponding to a cube side
	templateSidePoints = []
	templateSidePoints.append(templatePts)
	templateSidePoints.append(templatePts)

	targetSidePoints = []
	targetSidePoints.append(targetPts1)
	targetSidePoints.append(targetPts2)

	# Coordinates of the Cube Corners in the template image
	c1 = numpy.array([0,0,1])
	c2 = numpy.array([218,0,1])
	c3 = numpy.array([218,218,1])
	c4 = numpy.array([0,218,1])

	cornerCoords = numpy.array([c1,c2,c3,c4])

	# Find the Corners relating to the target points in the template image
	sideCorners = [];	sideHomography = []
	for i in range(0,2):
		corners, homography = getSquareCorners(cornerCoords,templateSidePoints[i],targetSidePoints[i])
		sideCorners.append(corners)
		sideHomography.append(homography)

	# Mark The Corresponding Internal Points for Calculating Homography
	for j in range(0,2):
		for i in range(0,4):
			templateCoord = (int(templateSidePoints[j][0,i]),int(templateSidePoints[j][1,i]))
			cv.Circle(templateImg, templateCoord, 2, (0,0,255),-1)
			cv.PutText(templateImg, str(i), templateCoord, FONT, (0,255,255))

			targetCoord = (int(targetSidePoints[j][0,i]),int(targetSidePoints[j][1,i]))
			cv.Circle(targetImg, targetCoord, 2, (0,0,255),-1)
			cv.PutText(targetImg, str(i), targetCoord, FONT, (0,255,255))


	# View the Images
	for i in range(0,2):
		for j in range(0,4):
			coord = (sideCorners[i][j,0],sideCorners[i][j,1])
			cv.Circle(targetImg, coord, 10, (255,0,0))
#			cv.PutText(targetImg, str(j), coord, FONT, (0,255,255))

	# Find the shared edge points
	cornerCoords = findAxialPoints(sideCorners[0],sideCorners[1])
	
	# Mark Axis Corners for Pose Estimation
	for cornerCoord in cornerCoords:
		cv.Circle(targetImg, cornerCoord, 3, (255,255,0),-1)
	cv.PutText(targetImg, 'x', cornerCoords[1], FONT, (0,255,255))
	cv.PutText(targetImg, 'y', cornerCoords[2], FONT, (0,255,255))
	cv.PutText(targetImg, 'z', cornerCoords[3], FONT, (0,255,255))


	cv.ShowImage("Template Image", templateImg)
	cv.MoveWindow("Template Image", 60, 0)
	cv.ShowImage("Target Image with Corners marked", targetImg)
	cv.MoveWindow("Target Image with Corners marked", 640+60, 0)
	cv.WaitKey()


	# Determine Pose from 4 corners
	objPts = [(0,0,0),(1,0,0),(0,1,0),(0,0,1)]	# Points used for pose estimation
	xtrPts = [(1,1,0),(1,0,1),(0,1,1),(1,1,1)]	# Other points on the cube, for display
	poseMat = findObjPose(560,cornerCoords)

	# Verify Estimation
	img = cv.LoadImageM(targetPath, cv.CV_LOAD_IMAGE_COLOR)
	img = markProjectedCube(img,cameraMatrix,poseMat)

	# Mark Original Points for Pose Calculation
	for i in range(0,4):
		cv.Circle(img, cornerCoords[i], 2, (255,0,255),-1)
		
	cv.ShowImage("Corners",img)
	cv.MoveWindow("Corners", 60, 0)
	cv.WaitKey()

	# Close Windows from previous demos
	#cv.DestroyAllWindows()
	return

def markProjectedCube(img,cameraMatrix,poseMat):
#	markedImg = cv.CreateMat(img.height,img.width,cv.CV_8UC3)
	
	objPts = [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]

	# Project Object Points to Image Plane
	imageCoords = projectPoints(objPts,cameraMatrix,poseMat)

	# Mark Cube Corners
	cv.Circle(img, imageCoords[0], 10, (255,255,0))	# Reference Point
	cv.Circle(img, imageCoords[1], 10, (0,0,255))	# +x
	cv.Circle(img, imageCoords[2], 10, (0,255,0))	# +y
	cv.Circle(img, imageCoords[3], 10, (255,0,0))	# +z
	cv.Circle(img, imageCoords[4], 10, (0,255,255))
	cv.Circle(img, imageCoords[5], 10, (0,255,255))
	cv.Circle(img, imageCoords[6], 10, (0,255,255))

	cv.Line(img, imageCoords[0], imageCoords[1], (255,255,255))
	cv.Line(img, imageCoords[0], imageCoords[2], (255,255,255))
	cv.Line(img, imageCoords[0], imageCoords[3], (255,255,255))
	cv.Line(img, imageCoords[4], imageCoords[1], (255,255,255))
	cv.Line(img, imageCoords[4], imageCoords[2], (255,255,255))
	cv.Line(img, imageCoords[5], imageCoords[1], (255,255,255))
	cv.Line(img, imageCoords[5], imageCoords[3], (255,255,255))
	cv.Line(img, imageCoords[6], imageCoords[2], (255,255,255))
	cv.Line(img, imageCoords[6], imageCoords[3], (255,255,255))

	return img

def findAxialPoints(sideCorners1,sideCorners2):
	# Finds the 4 non-coplaner points used by POSIT given
	# the (x,y) coordinates of the corners of 4 intersecting sides

	# Determine the 2 pairs of points that are closest together
	sDist = ()		# Smallest Distance
	ssDist = ()		# 2nd Smallest Distance
	sID = 0			# Id of the pair with the smallest distance
	ssID = 0		# Id of the pair with the 2nd smallest distance

	for i in range(0,4):
		for j in range(i,4):
			p1 = (sideCorners1[i,0],sideCorners1[i,1])
			p2 = (sideCorners2[j,0],sideCorners2[j,1])
			d = pixelDist(p1,p2)

			if (d < sDist):
				ssDist = sDist
				ssID = sID
				sDist = d
				sID = (i,j)
			else:
				if (d < ssDist):
					ssDist = d
					ssID = (i,j)

	# Determine the ordering based on the indices of the first square
	if ((ssID[0] - sID[0])%4) == 1:
		c1 = ((sID[0]-1)%4,sID[0],ssID[0])
		c2 = ((sID[1]+1)%4,sID[1],ssID[1])
	else:
		c1 = ((ssID[0]-1)%4,ssID[0],sID[0])
		c2 = ((ssID[1]+1)%4,ssID[1],sID[1])

	cornerCoords = []
	# '(0,0,0)' point
	avgX = (sideCorners1[c1[1],0] + sideCorners2[c2[1],0])/2
	avgY = (sideCorners1[c1[1],1] + sideCorners2[c2[1],1])/2
	p = (avgX,avgY)
	cornerCoords.append(p)
	# 'x' Axis point
	p = (sideCorners1[c1[0],0],sideCorners1[c1[0],1])
	cornerCoords.append(p)
	# 'y' Axis point
	p = (sideCorners2[c2[0],0],sideCorners2[c2[0],1])
	cornerCoords.append(p)
	# 'z' Axis point
	avgX = (sideCorners1[c1[2],0] + sideCorners2[c2[2],0])/2
	avgY = (sideCorners1[c1[2],1] + sideCorners2[c2[2],1])/2
	p = (avgX,avgY)
	cornerCoords.append(p)

	# Return list of (x,y) coordinates
	return cornerCoords

def combinePoseMats(rotMat,tvec):
	# Assumes rotMat and tvec are cvMatrices
	poseMat = cv.CreateMat(3,4,cv.CV_32FC1)
	poseMat[0,0] = rotMat[0,0]; poseMat[0,1] = rotMat[0,1]; poseMat[0,2] = rotMat[0,2]
	poseMat[1,0] = rotMat[1,0]; poseMat[1,1] = rotMat[1,1]; poseMat[1,2] = rotMat[1,2]
	poseMat[2,0] = rotMat[2,0]; poseMat[2,1] = rotMat[2,1]; poseMat[2,2] = rotMat[2,2]

	poseMat[0,3] = tvec[0,0]
	poseMat[1,3] = tvec[1,0]
	poseMat[2,3] = tvec[2,0]

	return poseMat

def eulerAngles2RotMat(rvec):
	# http://en.wikipedia.org/wiki/Euler_angles#Matrix_orientation
	# rvec is a 3x1 cvMat
	s = [0, 0, 0]
	s[0] = math.sin(rvec[0,0])
	s[1] = math.sin(rvec[1,0])
	s[2] = math.sin(rvec[2,0])
	c = [0, 0, 0]
	c[0] = math.cos(rvec[0,0])
	c[1] = math.cos(rvec[1,0])
	c[2] = math.cos(rvec[2,0])

	rotMat = cv.CreateMat(3,3,cv.CV_32FC1)
	rotMat[0,0] = c[1]*c[2]
	rotMat[0,1] = -c[1]*s[2]
	rotMat[0,2] = s[1]
	rotMat[1,0] = c[0]*s[2] + c[2]*s[0]*s[1]
	rotMat[1,1] = c[0]*s[2] - s[0]*s[1]*s[2]
	rotMat[1,2] = -c[1]*s[0]
	rotMat[2,0] = s[0]*s[2] - c[0]*c[2]*s[1]
	rotMat[2,1] = c[2]*s[0] - c[0]*s[1]*s[2]
	rotMat[2,2] = c[0]*c[1]

	return rotMat

def rotMat2EulerAngles(rotationMatrix):
	# rotationMatrix is a 3x3 cvMat
	# http://en.wikipedia.org/wiki/Rotation_matrix#Determining_the_angle
	phi	= math.atan2(rotationMatrix[2,0],rotationMatrix[2,1])
	theta = math.acos(rotationMatrix[2,2])
	psi = - math.atan2(rotationMatrix[0,2],rotationMatrix[1,2])

	return (phi,theta,psi)


def pixelDist(p1,p2):
	dx = p1[0] - p2[0]
	dy = p1[1] - p2[1]
	d = math.sqrt(dx*dx + dy*dy)
	return d

def findPlanarPose(homography,cameraMatrix='default'):
	# Doesnt seem to work properly
	# http://urbanar.blogspot.com/2011/04/from-homography-to-opengl-modelview.html
	# Estimates rotation matrix for a plane based on its homography transform
	# and the intrinsic camera matrix

	if (cameraMatrix == 'default'):
		cameraMatrix = cv.CreateMat(3,3,cv.CV_32FC1)
		cv.SetZero(cameraMatrix)
		cameraMatrix[0,0] = 560.0
		cameraMatrix[0,2] = 300.0
		cameraMatrix[1,1] = 560.0
		cameraMatrix[1,2] = 230.0
		cameraMatrix[2,2] = 1.0

	inverseCameraMatrix = cv.CreateMat(3,3,cv.CV_32FC1)
	inverseCameraMatrix[0,0] = 1.0/cameraMatrix[0,0]
	inverseCameraMatrix[0,1] = 0.0
	inverseCameraMatrix[0,2] = -cameraMatrix[0,2]/cameraMatrix[0,0]
	inverseCameraMatrix[1,0] = 0.0
	inverseCameraMatrix[1,1] = 1.0/cameraMatrix[1,1]
	inverseCameraMatrix[1,2] = -cameraMatrix[1,2]/cameraMatrix[1,1]
	inverseCameraMatrix[1,0] = 0.0
	inverseCameraMatrix[1,1] = 0.0
	inverseCameraMatrix[1,2] = 1.0

	h1 = cv.CreateMat(3,1,cv.CV_32FC1)
	h2 = cv.CreateMat(3,1,cv.CV_32FC1)
	h3 = cv.CreateMat(3,1,cv.CV_32FC1)
	inverseH1 = cv.CreateMat(3,1,cv.CV_32FC1)

	h1[0,0] = homography[0,0]
	h1[1,0] = homography[1,0]
	h1[2,0] = homography[2,0]

	h2[0,0] = homography[0,1]
	h2[1,0] = homography[1,1]
	h2[2,0] = homography[2,1]

	h3[0,0] = homography[0,2]
	h3[1,0] = homography[1,2]
	h3[2,0] = homography[2,2]

	cv.GEMM(inverseCameraMatrix, h1, 1.0, None, 0.0, inverseH1)
	lmda = numpy.sqrt(h1[0,0]*h1[0,0] + h1[1,0]*h1[1,0] + h1[2,0]*h1[2,0])

	if(lmda != 0):
		lmda = 1.0/lmda
		inverseCameraMatrix[0,0] *= lmda
		inverseCameraMatrix[1,0] *= lmda
		inverseCameraMatrix[2,0] *= lmda
		inverseCameraMatrix[0,1] *= lmda
		inverseCameraMatrix[1,1] *= lmda
		inverseCameraMatrix[2,1] *= lmda
		inverseCameraMatrix[0,2] *= lmda
		inverseCameraMatrix[1,2] *= lmda
		inverseCameraMatrix[2,2] *= lmda

	r1 = cv.CreateMat(3,1,cv.CV_32FC1)
	r2 = cv.CreateMat(3,1,cv.CV_32FC1)
	r3 = cv.CreateMat(3,1,cv.CV_32FC1)
	cv.GEMM(inverseCameraMatrix, h1, 1.0, None, 0.0, r1)
	cv.GEMM(inverseCameraMatrix, h2, 1.0, None, 0.0, r2)
	cv.CrossProduct(r1,r2,r3)

    # Put rotation columns into rotation matrix... with some unexplained sign changes
	rotationMatrix = cv.CreateMat(3,3,cv.CV_32FC1)
	rotationMatrix[0,0] = r1[0,0]
	rotationMatrix[0,1] = -r2[0,0]
	rotationMatrix[0,2] = -r3[0,0]
	rotationMatrix[1,0] = -r1[1,0]
	rotationMatrix[1,1] = r2[1,0]
	rotationMatrix[1,2] = r3[1,0]
	rotationMatrix[2,0] = -r1[2,0]
	rotationMatrix[2,1] = r2[2,0]
	rotationMatrix[2,2] = r3[2,0]

	translationVector = cv.CreateMat(3,1,cv.CV_32FC1)
	cv.GEMM(inverseCameraMatrix, h3, 1.0, None, 0.0, translationVector)
	translationVector[0,0] *= 1
	translationVector[1,0] *= -1
	translationVector[2,0] *= -1
	w = cv.CreateMat(3,3,cv.CV_32FC1)
	ut = cv.CreateMat(3,3,cv.CV_32FC1)
	vt = cv.CreateMat(3,3,cv.CV_32FC1)
	cv.SVD(rotationMatrix,w,ut,vt,cv.CV_SVD_V_T + cv.CV_SVD_U_T)

	cv.GEMM(ut, vt, 1.0, None, 0.0, rotationMatrix)

	return rotationMatrix, translationVector

def demoPlanerPoseEstimation():
	# This does not work well.
	templatePath = "images/redCubeFront.png"
	targetPath = "images/redCubeTurned.png"
	templateImg = cv.LoadImageM(templatePath, cv.CV_LOAD_IMAGE_COLOR)
	targetImg = cv.LoadImageM(targetPath, cv.CV_LOAD_IMAGE_COLOR)

	# Four of the internal points on the test images
	templatePts = cv.CreateMat(2,4,cv.CV_32FC1)
	targetPts = cv.CreateMat(2,4,cv.CV_32FC1)
#
	templatePts[0,0] = 28.0
	templatePts[1,0] = 16.0

	templatePts[0,1] = 54.0
	templatePts[1,1] = 26.0

	templatePts[0,2] = 43.0
	templatePts[1,2] = 51.0

	templatePts[0,3] = 18.0
	templatePts[1,3] = 41.0
#
	targetPts[0,0] = 62.0
	targetPts[1,0] = 22.0

	targetPts[0,1] = 80.0
	targetPts[1,1] = 33.0

	targetPts[0,2] = 73.0
	targetPts[1,2] = 56.0

	targetPts[0,3] = 55.0
	targetPts[1,3] = 45.0
#
	c1 = numpy.array([0,0,1])
	c2 = numpy.array([0,templateImg.width,1])
	c3 = numpy.array([templateImg.width,0,1])
	c4 = numpy.array([templateImg.width,templateImg.width,1])
	cornerCoords = numpy.array([c1,c2,c3,c4])
	corners, homography = getSquareCorners(cornerCoords,templatePts,targetPts)

	rotationMatrix, translationVector = findPlanarPose(homography)

	print "Rotation Matrix:"
	print rotationMatrix[0,0],rotationMatrix[0,1],rotationMatrix[0,2]
	print rotationMatrix[1,0],rotationMatrix[1,1],rotationMatrix[1,2]
	print rotationMatrix[2,0],rotationMatrix[2,1],rotationMatrix[2,2]

	print "Translation Vector"
	print translationVector[0,0]
	print translationVector[1,0]
	print translationVector[2,0]

	eulerAngles = rotMat2EulerAngles(rotationMatrix)
	print "Euler Angles (radians):"
	print eulerAngles

	poseMatrix = cv.CreateMat(3,4,cv.CV_32FC1)
	poseMatrix[:,0:2] = rotationMatrix
#	poseMatrix[:,3] = translationVector

	print "Pose Matrix"
	print poseMatrix[0,0],poseMatrix[0,1],poseMatrix[0,2],poseMatrix[0,3]
	print poseMatrix[1,0],poseMatrix[1,1],poseMatrix[1,2],poseMatrix[1,3]
	print poseMatrix[2,0],poseMatrix[2,1],poseMatrix[2,2],poseMatrix[2,3]

	return

if __name__ == '__main__':
	try:
		active = True
		while active:
			print "Select from the following demos:"
			print "c: Premade Camera Calibration Demo"
			print "h: Cube Corner Homography Demo"
			print "p: Posit Demo"
			print "e: Pose Estimation Chain Demo"
			print "q: Quit"

			c = raw_input()

			if c.strip() == 'c':
				cameraCalibration()
			elif c.strip() == 'h':
				demoCubeSideCorners()
				#demoSquareCorners()
			elif c.strip() == 'p':
				positDemo()
			elif c.strip() == 'e':
				demoPoseEstimationChain()
			elif c.strip() == 'q':
				active = False
			else:
				print "No valid option selected."

	except rospy.ROSInterruptException: 
		pass

