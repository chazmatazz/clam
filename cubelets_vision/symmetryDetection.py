#!/usr/bin/env python
#import roslib; roslib.load_manifest('cubelets_vision')
import roslib; roslib.load_manifest('beginner_tutorials')
import sys
import rospy
import cv,cv2
import numpy
import math,random
from scipy.spatial import KDTree
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError


def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def getFeatures(img, hessThresh=500):
	#""" get features of image """
	return cv.ExtractSURF(img, None, cv.CreateMemStorage(), (0, hessThresh, 3, 1))


def findRotatedKeyPoints(imgPath):
	img = cv.LoadImageM(imgPath, cv.CV_LOAD_IMAGE_GRAYSCALE)
	cv.Smooth(img, img, cv.CV_GAUSSIAN, 5, -1)
	(keyPoints, descriptors) = getFeatures(img)

	w = img.width
	h = img.height
	# bew = border exclusion width
	bew = 30

	# Calculate centers of rotation for each pair of keypoints
	pairCenters = findRotationalPairs(keyPoints,w,h,bew)

	# Cluster centers of rotation and determine inliers
	clusterInliers,clusterCenters = clusterCenterInliers(pairCenters)

	isAdded = []
	for i in range(0,len(keyPoints)):
		isAdded.append(0)

	inlierGroups = []
	for i in range(0,len(clusterCenters)):
		inlierKpts = []
		for (kpt1,kpt2,center) in clusterInliers:
			if (isAdded[kpt1] == 0):
				inlierKpts.append(keyPoints[kpt1])
				isAdded[kpt1] = 1
			if (isAdded[kpt2] == 0):
				inlierKpts.append(keyPoints[kpt2])
				isAdded[kpt1] = 1

		inlierGroups.append((clusterCenters[i],inlierKpts))

	return inlierGroups


def findRotationalPairs(keyPoints,w,h,bew):
	# For use with surf keypoints
	# Calculate center of rotation for each pair of points
	centers = cv.CreateMat(h,w,cv.CV_8UC1)
	cv.SetZero(centers)
	#
	pairCenters = []
	# bew = border exclusion width
	bew = 30
	for i in range(0,len(keyPoints)):
		for j in range(i+1,len(keyPoints)):
			if  (keyPoints[i][3] < keyPoints[j][3]):
				pi = keyPoints[i]
				pj = keyPoints[j]
			else:
				pj = keyPoints[i]
				pi = keyPoints[j]

			((xi, yi), laplaciani, sizei, diri, hessiani) = pi
			((xj, yj), laplacianj, sizej, dirj, hessianj) = pj

			# Check if solution exists, then if size of keypoints roughly the same
			if (sizei - sizej > -180)&(min((sizei,sizej))/max((sizei,sizej)) > 0.98):
				# beta in degrees
				print "Dir of keypoints: " , round(diri),round(dirj)
				beta = ((diri - dirj) + 180.0) / 2.0 * (math.pi / 180)
				print "beta: " , beta
				# get gamma in radians
				xAxisVec = (1,0)
				gamma = angle((xj-xi,yj-yi), xAxisVec)
				print "gamma: " , gamma

				dx = xj - xi;	dy = yj - yi
				d = numpy.sqrt(dx*dx + dy*dy)
				r = d * numpy.sqrt(1 + math.tan(gamma)*math.tan(gamma)) / 2.0
			
				cx = int(xi + r*numpy.cos(beta + gamma))
				cy = int(yi - r*numpy.sin(beta + gamma))
			
				# Check if center is within the image (and not in border exclusion zone)
				if (cx>0+bew)&(cy>0+bew)&(cx<w-bew)&(cy<h-bew)&(r<(w+h)/2.0):
					print "radius: " , r
					print "cx,cy ", (cx,cy)
					centers[cy,cx] += 63
					centers[cy-1,cx-1] += 31
					centers[cy-1,cx+1] += 31
					centers[cy+1,cx-1] += 31
					centers[cy+1,cx+1] += 31
					pairCenters.append((i,j,(cx,cy)))
	# Returns indices to keyPoints, as well as the center of rotation
	return pairCenters

def clusterCenterInliers(pairCenters,K=20,showImages=False,dims=(640,480)):
	(w,h) = dims

	kmeansInput = []
	for (i,j,(cx,cy)) in pairCenters:
		kmeansInput.append((cx,cy,0))
	clusterIDs = clusterByTriplet(kmeansInput,K)

	# Sort Clusters into Bins
	clusters = []
	for i in range(0,K):
		clusters.append([])

	for i in range(0,len(pairCenters)):
		clusters[int(clusterIDs[i,0])].append(pairCenters[i])

	# Display Clusters
	if (showImages):
		#example = cv.LoadImageM(path, cv.CV_LOAD_IMAGE_COLOR)
		example = cv.CreateMat(h,w, cv.CV_8UC3)
		cv.SetZero(example)
		for i in range(0,K):
			clusterColor = randColorTriplet()
			for (i,j,(cx,cy)) in clusters[i]:
				cv.Circle(example, (cx,cy), 3, clusterColor )
		cv.ShowImage("Centers",example)
		cv.WaitKey()

	# Determine Inliers
	clusterInliers = []
	clusterCenters = []
	for i in range(0,K):
		clusterInliers.append([])

	centerMap = cv.CreateMat(h,w,cv.CV_8UC1)

	for index in range(0,K):
		cv.SetZero(centerMap)

		for (i,j,(cx,cy)) in clusters[index]:
			centerMap[cy,cx] += 100
			centerMap[cy-1,cx] += 50
			centerMap[cy+1,cx] += 50
			centerMap[cy,cx+1] += 50
			centerMap[cy,cx-1] += 50

		# Find Center of Mass
		totalMass = 0
		avgX = 0
		avgY = 0
		for x in range(0,w):
			for y in range(0,h):
				totalMass += centerMap[y,x]
				avgX += x * centerMap[y,x]
				avgY += y * centerMap[y,x]
		if (totalMass > 0):
			avgX = int(avgX / totalMass)
			avgY = int(avgY / totalMass)
		else:
			avgX = 0
			avgY = 0


		# Display Clusters
		if (showImages):
			#example = cv.LoadImageM(path, cv.CV_LOAD_IMAGE_COLOR)
			example = cv.CreateMat(h,w, cv.CV_8UC3)
			cv.SetZero(example)
			for (i,j,(cx,cy)) in clusters[index]:
				cv.Circle(example, (cx,cy), 3, clusterColor )
			cv.Circle(example, (avgX,avgY), 15, (255,0,0) )
			print "Cluster Center: ", (avgX,avgY)
			cv.ShowImage("Center",example)
			cv.WaitKey()

		clusterCenters.append((int(avgX),int(avgY)))

		# Determine Inliers
		for (i,j,(cx,cy)) in clusters[index]:
			dx = cx - avgX
			dy = cy - avgY
			d = math.sqrt(dx*dx + dy*dy)
			if (d < 15):
				clusterInliers[index].append((i,j,(cx,cy)))

	# Remove Clusters with fewer than 4 inliers
	indexList = range(0,K)
	indexList.reverse()
	for index in indexList:
		print "index: ", index
		if (len(clusterInliers[index]) < 4):
			print "clusterRemoved ", index
			clusterInliers.pop(index)
			clusterCenters.pop(index)

	# Return a list of clusters, with each cluster being a list of (i,j,(cx,cy)) center pairs
	return clusterInliers,clusterCenters

def clusterSymmetryDemo(path="images/cubeInScene.png", K = 20):
	example = cv.LoadImageM(path, cv.CV_LOAD_IMAGE_GRAYSCALE)
	cv.Smooth(example, example, cv.CV_GAUSSIAN, 5, -1)
	(keyPoints, descriptors) = getFeatures(example)

	w = example.width
	h = example.height
	# bew = border exclusion width
	bew = 30

	# Calculate centers of rotation for each pair of keypoints
	print "Caluclating Centers of Rotation"
	pairCenters = findRotationalPairs(keyPoints,w,h,bew)

	# Cluster centers of rotation and determine inliers
	print "Determining Inliers"
	clusterInliers,clusterCenters = clusterCenterInliers(pairCenters)

	# Display Inliers
	example = cv.LoadImageM(path, cv.CV_LOAD_IMAGE_COLOR)
	print "Inliers: "
	for index in range(0,len(clusterInliers)):
		clusterColor = randColorTriplet()
		for (i,j,(cx,cy)) in clusterInliers[index]:
			cv.Circle(example, (cx,cy), 3, clusterColor )
		# Mark Centers
		cv.Circle(example, clusterCenters[index], 15, (255,255,255) )
		print "Cluster Center: ", clusterCenters[index]
	print ""
	print "Rotational Clusters Found: ", len(clusterInliers)
	cv.ShowImage("Inliers",example)
	cv.WaitKey()
	
def overlayKeyPoint(imgMat,keyPoint,color,offset=(0,0)):
	# Overlay a surf feature marker onto an image
	# If grayscale, convert to color
	if isinstance(imgMat[0,0], float):
		overlaid = cv.CreateMat(imgMat.height, imgMat.width, cv.CV_8UC3)
		cv.CvtColor(imgMat, overlaid, cv.CV_GRAY2BGR)
	else:
		overlaid = imgMat
	
	((x, y), laplacian, size, fdir, hessian) = keyPoint
	r = int(0.5*size)
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
		r = int(0.5*size)
		px = int(x + offset[0])
		py = int(y + offset[1])
		cv.Circle(overlaid, (px, py), r, color)
		radians = fdir/180.0*math.pi
		rpx = int(px+r*math.cos(radians))
		rpy = int(py-r*math.sin(radians))
		cv.Line(overlaid, (px, py), (rpx, rpy), color)
	return overlaid

def clusterByTriplet(criteriaList,K,n=10):
	# criteriaList is a python list, K is the number of clusters to find
	triplets = cv.CreateMat(len(criteriaList), 1, cv.CV_32FC3)
	for i in range(0,len(criteriaList)):
	    triplets[i,0] = criteriaList[i]

	samples = cv.CreateMat(triplets.height, 1, cv.CV_32FC3)
	cv.Scale(triplets, samples)
	labels = cv.CreateMat(triplets.height, 1, cv.CV_32SC1)

	# n iterations of the K-means algorithm.
	crit = (cv.CV_TERMCRIT_EPS + cv.CV_TERMCRIT_ITER, n, 1.0)
	cv.KMeans2(samples, K, labels, crit)
	return labels

def randColorTriplet():
	color = (numpy.random.randint(0,255),numpy.random.randint(0,255),numpy.random.randint(0,255))
	return color

if __name__ == '__main__':
	try:
		clusterSymmetryDemo()
	except rospy.ROSInterruptException:
		pass

