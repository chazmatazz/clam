#!/usr/bin/env python
#import roslib; roslib.load_manifest('cubelets_vision')
import roslib; roslib.load_manifest('beginner_tutorials')
import sys
import rospy
import cv,cv2
import numpy
import math
from scipy.spatial import KDTree
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from cubeletsSURF2 import *

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
    # source, target should be 2x4+ CV_32FC1 matrices,
	# specifying the  corresponding2D coordinates of the points in the template and target image
	# sideLength is the length in pixels of the square's template image
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


def demoPlanerPoseEstimation():
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




def findPlanarPose(homography,cameraMatrix='default'):
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

def findObjPose(objPts,imgPts,focal_length):
	# objPts: CvPoint3D32fs, coordinates of points on object in object space
	# imgPts: CvPoint2D32f, object points projections on the 2D image plane
	positObject = cv.CreatePOSITObject(objPts)
	rvecs = cv.CreateMat(1,3,cv.CV_32FC1)
	tvecs = cv.CreateMat(1,3,cv.CV_32FC1)

	rvecs, tvecs = cv.POSIT(positObject,imgPts,focal_length,(cv.CV_TERMCRIT_ITER, 10, 0))

	# Memory Management
	cv.ReleasePOSITObject(positObject)

	return rvecs,tvecs

def demoFindObjPose():
	img = cv.LoadImageM("images/redCubeTurned.png", cv.CV_LOAD_IMAGE_COLOR)

	cameraMatrix = cv.CreateMat(3,3,cv.CV_32FC1)
	cv.SetZero(cameraMatrix)
	cameraMatrix[0,0] = 560.0
	cameraMatrix[0,2] = 300.0
	cameraMatrix[1,1] = 560.0
	cameraMatrix[1,2] = 230.0
	cameraMatrix[2,2] = 1.0

	distCoeffs = cv.CreateMat(4,1,cv.CV_32FC1)
	distCoeffs[0,0] = 0.245718212181
	distCoeffs[1,0] = -0.603892493849
	distCoeffs[2,0] = 0.00312595426646
	distCoeffs[3,0] = 0.0056193962696

	cubeSize = 1.0
	c1 = [0.0, 0.0, 0.0]
	c2 = [cubeSize, 0.0, 0.0]
	c3 = [0.0, cubeSize, 0.0]
	c4 = [0.0, 0.0, cubeSize]
	objectPts = numpy.array([c1,c2,c3,c4])
	print objectPts
	objectPts = cv.fromarray(objectPts)

	im1 = [37.0,70.0]
	im2 = [0.0,69.0]
	im3 = [38.0,5.0]
	im4 = [84.0,71.0]
	imgPoints = numpy.array([im1,im2,im3,im4])
	print imgPoints
	imgPoints = cv.fromarray(imgPoints)

	rvec = cv.CreateMat(3,1,cv.CV_32FC1)
	tvec = cv.CreateMat(3,1,cv.CV_32FC1)

	cv.FindExtrinsicCameraParams2(objectPts, imgPoints, cameraMatrix, distCoeffs, rvec, tvec)
	print "rvec"
	print rvec[0,0]
	print rvec[1,0]
	print rvec[2,0]

	print "tvec"
	print tvec[0,0]
	print tvec[1,0]
	print tvec[2,0]

	rotMat = eulerAngles2RotMat(rvec)
	poseMat = combinePoseMats(rotMat,tvec)

	point = cv.CreateMat(4,1,cv.CV_32FC1)
	pointCSpace = cv.CreateMat(3,1,cv.CV_32FC1)
	uv1 = cv.CreateMat(3,1,cv.CV_32FC1)
	for c in (c1,c2,c3,c4):
		point[0,0] = c[0]
		point[1,0] = c[1]
		point[2,0] = c[2]
		point[3,0] = 1.0
		print c
		cv.GEMM(poseMat, point, 1.0, None, 0.0, pointCSpace)
		cv.GEMM(cameraMatrix, pointCSpace, 1.0, None, 0.0, uv1)

		print int(uv1[0,0]),int(uv1[1,0])
		cv.Circle(img, (int(uv1[0,0]),int(uv1[1,0])), 3, (255,0,0))

	cv.ShowImage("Corners",img)
	cv.WaitKey()


def combinePoseMats(rotMat,tvec):
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

if __name__ == '__main__':
	try:
		cameraCalibration()
		#demoFindObjPose()
		#demoPlanerPoseEstimation()
		#demoSquareCorners()
		#inliers = getInliers()
		#for i in range(0,len(inliers)):
		#	print inliers[i][0]
		#	inlierList = keyPts2CoordMat(inliers[i][1])
	except rospy.ROSInterruptException: 
		pass

