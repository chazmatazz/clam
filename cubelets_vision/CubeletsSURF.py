""" Library for Cubelet SURF """

import roslib; roslib.load_manifest('cubelets_vision')
import sys
import rospy
import cv,cv2
import numpy
import math
from scipy.spatial import KDTree
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

BLACK_CUBE = "images/black_cube.png"
CUBE_ARRAY = "images/synthetic.png"#"images/cubeArray.png"
IMAGE_WINDOW_NAME = "Image"
WEBCAM_WINDOW_NAME = "WebCam"
UPPER_LEFT_IMAGE = "Upper Left"
CONVOLVED_UPPER_LEFT_IMAGE = "Convolved Upper Left"
HASH_CIRCLES = "Hash Circles"
VOTING_IMAGE = "Upper Left Voting"
CAMERA_INDEX = 0
MATCH_SIZE = 5
INLIER_SIZE = 10
BRIGHTENED_IMAGE = "Brightened Image"
FONT = cv.InitFont(cv.CV_FONT_HERSHEY_PLAIN, 1, 1, 0, 3, 8)

def loadImage(path, type=cv.CV_LOAD_IMAGE_GRAYSCALE):
    """ load an image """
    return cv.LoadImageM(path, type)

def getFeatures(img, hessThresh=1000):
    """ get features of image """
    return cv.ExtractSURF(img, None, cv.CreateMemStorage(), (0, hessThresh, 3, 1))

def getMultiMatchFeatures(template, test, descriptor_radius):
    """ match features by distance
    return list of features (templateFeatureIndex, testFeatureIndices) """
    (templateKeypoints, templateDescriptors) = template
    (testKeypoints, testDescriptors) = test
    kdtree = KDTree(testDescriptors)
    results = []
    for i in range(len(templateDescriptors)):
        results += [(i, kdtree.query_ball_point(templateDescriptors[i], descriptor_radius))]
    return results

def getSingleMatchFeatures(template, test, thresh=0.75):
    """ Chen's code """
    (templateKeypoints, templateDescriptors) = getFeatures(template)
    (testKeypoints, testDescriptors) = getFeatures(test)
    kdtree = KDTree(testDescriptors)
    m1 = []
    m2 = []

    for i in range(len(templateDescriptors)):
	(dist,index) = kdtree.query(templateDescriptors[i],2)
	if ((dist[0] / dist[1]) < thresh):
	    k1 = templateKeypoints[i]
	    k2 = testKeypoints[index[0]]

	    m1.append(k1)
	    m2.append(k2)
    return m1,m2

def getStableFeatures(frame1,frame2,thresh=0.75):
    # Find stable SURF features between two frames
    (f1Keypoints, f1Descriptors) = getFeatures(frame1)
    (f2Keypoints, f2Descriptors) = getFeatures(frame2)
    kdtree = KDTree(f2Descriptors)

    stableKeypoints = []
    stableDescriptors = []
    for i in range(len(f1Descriptors)):
	(dist,index) = kdtree.query(f1Descriptors[i],2)
	if ((dist[0] / dist[1]) < thresh):
	    stableKeypoints.append(f2Keypoints[index[0]])
	    stableDescriptors.append(f2Descriptors[index[0]])
    return stableKeypoints,stableDescriptors

def rotationMatrix2x2(radians):
    sina = math.sin(radians)
    cosa = math.cos(radians)
    return numpy.matrix([[cosa, sina], [-sina, cosa]])

def getInliers(templateImagePath=BLACK_CUBE, testImagePath=CUBE_ARRAY, descriptor_radius=0.1):
    templateImage = loadImage(templateImagePath)
    testImage = loadImage(testImagePath)
    
    template = getFeatures(templateImage)
    test = getFeatures(testImage)
    imSize = cv.GetSize(testImage)
    (width, height) = imSize

    matchedFeatures = getMultiMatchFeatures(template, test, descriptor_radius)

    matches = findMatches(imSize, testImage, template, test)
    upper_lefts = getUpperLefts(matchedFeatures, template, test)

    myMatchedImage = cv.CreateMat(height, width, cv.CV_32FC1)
    zeroC1(myMatchedImage)

    (templateKeypoints, templateDescriptors) = template

    inliers = []
    for match in matches:
        currentInliers = []
        points =[]
        cv.Circle(myMatchedImage, match, INLIER_SIZE, 255, -1)
        for upper_left in upper_lefts:
            (i, k, p) = upper_left
            x = round(p[0])
            y = round(p[1])
            if 0 <= x and x < width and 0 <= y and y < height:
                if myMatchedImage[y,x] == 255:
                    currentInliers += [upper_left]
        for node in currentInliers:
            (index, keyPoint, upperLeft) = node
            #print node
            
            points += [(keyPoint,templateKeypoints[index])]
        inliers += [(match,points)]
        cv.Circle(myMatchedImage, match, INLIER_SIZE, 0, -1)

    for i in inliers:
        print i
    cv.WaitKey()
    return inliers # in form of (inlier_point, array of keypoints that hit point)

def getUpperLefts(matchedFeatures, template, test):
	(templateKeypoints, templateDescriptors) = template
	(testKeypoints, testDescriptors) = test
	upper_lefts = []
	for (templateFeatureIndex, testFeatureIndices) in matchedFeatures:
		(templatePoint, templateLaplacian, templateSize, templateDirection, templateHessian) = templateKeypoints[templateFeatureIndex]
		for testFeatureIndex in testFeatureIndices:
			(testPoint, testLaplacian, testSize, testDirection, testHessian) = testKeypoints[testFeatureIndex]
            		# vote for upper left
			rot = rotationMatrix2x2(math.radians(testDirection - templateDirection))
			(x,y) = templatePoint
			offset = (x * rot[0,0] + y * rot[1,0], x * rot[0,1] + y * rot[1,1])
			featureUpperLeft = numpy.array(testPoint) - numpy.array(offset)# * 1.0 * testSize/templateSize
			upper_lefts += [(templateFeatureIndex, testKeypoints[testFeatureIndex], featureUpperLeft)]
	return upper_lefts

def zeroC1(img):
    imSize = cv.GetSize(img)
    (width, height) = imSize
    for i in range(height):
        for j in range(width):
            img[i,j] = 0

def findMatches(imSize, testImage, template, test, descriptor_radius=0.1, vote_radius=20, vote_threshold=0.03, exclusion_radius=75):
    """ match features to a template """
    # TODO annotate matches with features
    (width, height) = imSize
    matchedFeatures = getMultiMatchFeatures(template, test, descriptor_radius)
    
    (templateKeypoints, templateDescriptors) = template
    (testKeypoints, testDescriptors) = test

 #   for key in testKeypoints:
  #      print key
	
    # retrieve upper left in image space
    upper_lefts = getUpperLefts(matchedFeatures, template, test)
    
    # create a image for showing matched features and upper left voting
    voteImage = cv.CreateMat(height, width, cv.CV_8UC3)
    cv.CvtColor(testImage, voteImage, cv.CV_GRAY2BGR)
    
    class ColorIndex:
        """ Creates a set of evenly spaced color bytes """
        def __init__(self, length):
            self.length = length
        
        def getColor(self, idx):
            n = idx * 255 * 255 * 255 / self.length
            return ((n/(255*255))%255, (n/255)%255, n%255)

    def drawVote(img, color, testKeypoint, vote):
        (featurePoint, laplacian, size, direction, hessian) = testKeypoint
        p1 = (int(featurePoint[0]), int(featurePoint[1]))
        radians = math.radians(direction)
        p2 = (int(featurePoint[0]+size*math.cos(radians)), int(featurePoint[1]+size*math.sin(radians)))
        p3 = (int(vote[0]), int(vote[1]))
        cv.Circle(img, p1, size, color)
        cv.Line(img, p1, p2, color)
        cv.Line(img, p1, p3, color)
        
    colorIndex = ColorIndex(len(template[0]))
    for upper_left in upper_lefts:
        (templateFeatureIndex, testKeypoint, featureUpperLeft) = upper_left
        drawVote(voteImage, colorIndex.getColor(templateFeatureIndex), testKeypoint, featureUpperLeft)
    
    cv.ShowImage(VOTING_IMAGE, voteImage)
    
    # create a upper left image
    upperLeftImage = cv.CreateMat(height, width, cv.CV_32FC1)
    zeroC1(upperLeftImage)

    for upper_left in upper_lefts:
        (i, k, p) = upper_left
        x = round(p[0])
        y = round(p[1])
        if 0 <= x and x < width and 0 <= y and y < height:
            upperLeftImage[y, x] += 1
    
    cv.ShowImage(UPPER_LEFT_IMAGE, upperLeftImage)
    
    
    # find bright points in upper left image
    convolvedUpperLeftImage = cv.CreateMat(height, width, cv.CV_32FC1)
    cv.Smooth(upperLeftImage, convolvedUpperLeftImage, cv.CV_GAUSSIAN, vote_radius*2+1, -1)
    
    cv.ShowImage(CONVOLVED_UPPER_LEFT_IMAGE, convolvedUpperLeftImage)
    
    # get average value of convolved upper left image
    sum = 0
    for i in range(height):
        for j in range(width):
            sum += convolvedUpperLeftImage[i,j]
    
    brightenedImage = cv.CreateMat(height, width, cv.CV_32FC1)        
    for i in range(height):
        for j in range(width):
            brightenedImage[i, j] = convolvedUpperLeftImage[i, j] * testImage[i, j] * sum / (height * width) 
            
    cv.ShowImage(BRIGHTENED_IMAGE, brightenedImage)
    
    # find points in convolved upper left image above threshold
    aboveThreshold = []
    for i in range(height):
        for j in range(width):
            if convolvedUpperLeftImage[i,j] > vote_threshold:
                aboveThreshold += [(convolvedUpperLeftImage[i,j], (j,i))]

    aboveThresholdSorted = sorted(aboveThreshold, key=lambda elt: elt[0])
    aboveThresholdSorted.reverse()

    # find spaced points in upper left image
    hashImage = cv.CreateMat(height, width, cv.CV_8UC1) # 0 is candidate, not candidate otherwise
    zeroC1(hashImage)
    
    matches = []
    for elt in aboveThresholdSorted:
        p = elt[1]
        (x,y) = p
        if hashImage[y, x] == 0:
            matches += [p]
            cv.Circle(hashImage, p, exclusion_radius, 255, -1)
    cv.ShowImage(HASH_CIRCLES, hashImage)

    showMatchedImage(voteImage, matches)
    return matches    
        
def showMatchedImage(testImage, matches):
    for match in matches:
        cv.Circle(testImage, match, MATCH_SIZE, randColorTriplet(), -1)
    cv.ShowImage(IMAGE_WINDOW_NAME, testImage)

def matchTemplateImage(templateImagePath=BLACK_CUBE, testImagePath=CUBE_ARRAY):
    """ create a template, then match an image against it """
    templateImage = loadImage(templateImagePath)
    testImage = loadImage(testImagePath)
    
    template = getFeatures(templateImage)
    test = getFeatures(testImage)
    imSize = cv.GetSize(testImage)
    matches = findMatches(imSize, testImage, template, test)
    showMatchedImage(testImage, matches)
    cv.WaitKey()
    
def matchTemplateWebcam(templateImagePath=BLACK_CUBE, window_name=WEBCAM_WINDOW_NAME, camera_index=CAMERA_INDEX):
    """ create a template, then match webcam stream against it """
    cv.NamedWindow(window_name, cv.CV_WINDOW_AUTOSIZE)
    capture = cv.CaptureFromCAM(camera_index)
    
    template = getFeatures(loadImage(templateImagePath))

    #TODO

def randColorTriplet():
    """ create a random color """
    return (numpy.random.randint(0,255),numpy.random.randint(0,255),numpy.random.randint(0,255))

def cameraMatrixTest():
    cameraMatrix = cv.CreateMat(3,3,cv.CV_32FC1)
    cameraMatrix[0,0] = 1.0
    cameraMatrix[1,1] = 1.0
    cameraMatrix[0,1] = 0.0
    cameraMatrix[1,0] = 0.0
    cameraMatrix[0,2] = 0.0
    cameraMatrix[2,0] = 0.0
    cameraMatrix[2,2] = 1.0
    cameraMatrix[2,1] = 0.0
    cameraMatrix[1,2] = 0.0
    distCoeffs = cv.CreateMat(4,1,cv.CV_32FC1)
    rvecs = cv.CreateMat(1,3,cv.CV_32FC1)
    tvecs = cv.CreateMat(1,3,cv.CV_32FC1)
    p3 = cv.CreateMat(4,3,cv.CV_32FC1)
    p2 = cv.CreateMat(4,2,cv.CV_32FC1)
    pointCounts = cv.CreateMat(1,1,cv.CV_32SC1)
    pointCounts[0,0] = 4
    imageSize = (400,400)

#
    p3[0,0] = 98.0
    p3[0,1] = 92.0
    p3[0,2] = 0

    p3[1,0] = 309.0
    p3[1,1] = 118.0
    p3[1,2] = 0

    p3[2,0] = 324.0
    p3[2,1] = 250.0
    p3[2,2] = 0

    p3[3,0] = 121.0
    p3[3,1] = 327.0
    p3[3,2] = 0
#
    p2[0,0] = 157.0
    p2[0,1] = 128.0

    p2[1,0] = 317.0
    p2[1,1] = 98.0

    p2[2,0] = 311.0
    p2[2,1] = 192.0

    p2[3,0] = 126.0
    p2[3,1] = 305.0
    cv.CalibrateCamera2(p3, p2, pointCounts, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs,cv.CV_CALIB_USE_INTRINSIC_GUESS)
    #print cameraMatrix
	# try a prefound one.
    cameraMatrix[0,0] = 5.65146179e-01
    cameraMatrix[0,1] = 0.
    cameraMatrix[0,2] = 3.14276642e-01
    cameraMatrix[1,0] = 0.
    cameraMatrix[1,1] = 5.67260010e-01
    cameraMatrix[1,2] = 2.38061874e-01
    cameraMatrix[2,0] = 0.
    cameraMatrix[2,1] = 0.
    cameraMatrix[2,2] =1.

    return cameraMatrix

def findHomography(source,target):
    # source, target should be 2x4+ CV_32FC1 matrices
    homography = cv.CreateMat(3,3,cv.CV_32FC1)
    cv.FindHomography(source,target,homography)
    return homography

def getRotationMatrix(homography,cameraMatrix):
    # http://urbanar.blogspot.com/2011/04/from-homography-to-opengl-modelview.html
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

    h1[0,0] = p1_p2[0,0]
    h1[1,0] = p1_p2[1,0]
    h1[2,0] = p1_p2[2,0]

    h2[0,0] = p1_p2[0,1]
    h2[1,0] = p1_p2[1,1]
    h2[2,0] = p1_p2[2,1]

    h3[0,0] = p1_p2[0,2]
    h3[1,0] = p1_p2[1,2]
    h3[2,0] = p1_p2[2,2]

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
    rotationMatrix[0,0] = -r1[1,0]
    rotationMatrix[0,1] = r2[1,0]
    rotationMatrix[0,2] = r3[1,0]
    rotationMatrix[0,0] = -r1[2,0]
    rotationMatrix[0,1] = r2[2,0]
    rotationMatrix[0,2] = r3[2,0]

    translationVector = cv.CreateMat(3,1,cv.CV_32FC1)
    cv.GEMM(inverseCameraMatrix, h3, 1.0, None, 0.0, translationVector)
    translationVector[0,0] *= 1
    translationVector[1,0] *= -1
    translationVector[2,0] *= -1
    w = cv.CreateMat(3,3,cv.CV_32FC1)
    ut = rotationMatrix = cv.CreateMat(3,3,cv.CV_32FC1)
    vt = rotationMatrix = cv.CreateMat(3,3,cv.CV_32FC1)
    cv.SVD(rotationMatrix,w,ut,vt,cv.CV_SVD_V_T + cv.CV_SVD_U_T)

    cv.GEMM(ut, vt, 1.0, None, 0.0, rotationMatrix)

def homographyTest():
    # img1 = squareTest.png
    # img2 = squarePaper.png
    #
    p1 = cv.CreateMat(2,4,cv.CV_32FC1)
    p3 = cv.CreateMat(2,4,cv.CV_32FC1)
    p2 = cv.CreateMat(2,4,cv.CV_32FC1)
    p1_p2 = cv.CreateMat(3,3,cv.CV_32FC1)

    p1[0,0] = 98.0
    p1[1,0] = 92.0

    p1[0,1] = 309.0
    p1[1,1] = 118.0

    p1[0,2] = 324.0
    p1[1,2] = 250.0

    p1[0,3] = 121.0
    p1[1,3] = 327.0
#
    p2[0,0] = 157.0
    p2[1,0] = 128.0

    p2[0,1] = 317.0
    p2[1,1] = 98.0

    p2[0,2] = 311.0
    p2[1,2] = 192.0

    p2[0,3] = 126.0
    p2[1,3] = 305.0
    cv.FindHomography(p1,p2,p1_p2)
    #cv.ReleaseMat(p1)
    #cv.ReleaseMat(p2)
    c1 = numpy.array([17,14,1])
    c2 = numpy.array([384,14,1])
    c3 = numpy.array([17,384,1])
    c4 = numpy.array([384,384,1])
    c1 = numpy.dot(p1_p2,c1)
    c2 = numpy.dot(p1_p2,c2)
    c3 = numpy.dot(p1_p2,c3)
    c4 = numpy.dot(p1_p2,c4)
    c1 = (int(c1[0]/c1[2]), int(c1[1]/c1[2]))
    c2 = (int(c2[0]/c2[2]), int(c2[1]/c2[2]))
    c3 = (int(c3[0]/c3[2]), int(c3[1]/c3[2]))
    c4 = (int(c4[0]/c4[2]), int(c4[1]/c4[2]))

    im_in = cv.LoadImage("squareTest02.png",cv.CV_LOAD_IMAGE_COLOR)
    out = cv.CloneImage(im_in)
    cv.WarpPerspective(im_in, out, p1_p2)
    cv.Circle(im_in, c1, 10, (255,0,0))
    cv.Circle(im_in, c2, 10, (255,0,0))
    cv.Circle(im_in, c3, 10, (255,0,0))
    cv.Circle(im_in, c4, 10, (255,0,0))
    cv.ShowImage("Window", im_in)
    cv.WaitKey(0)
    cameraMatrix = cameraMatrixTest()

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
    inverseH1 = cv.CreateMat(3,1,cv.CV_32FC1)
    h2 = cv.CreateMat(3,1,cv.CV_32FC1)
    h3 = cv.CreateMat(3,1,cv.CV_32FC1)

    h1[0,0] = p1_p2[0,0]
    h1[1,0] = p1_p2[1,0]
    h1[2,0] = p1_p2[2,0]

    h2[0,0] = p1_p2[0,1]
    h2[1,0] = p1_p2[1,1]
    h2[2,0] = p1_p2[2,1]

    h3[0,0] = p1_p2[0,2]
    h3[1,0] = p1_p2[1,2]
    h3[2,0] = p1_p2[2,2]

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
    rotationMatrix[0,0] = -r1[1,0]
    rotationMatrix[0,1] = r2[1,0]
    rotationMatrix[0,2] = r3[1,0]
    rotationMatrix[0,0] = -r1[2,0]
    rotationMatrix[0,1] = r2[2,0]
    rotationMatrix[0,2] = r3[2,0]

    translationVector = cv.CreateMat(3,1,cv.CV_32FC1)
    cv.GEMM(inverseCameraMatrix, h3, 1.0, None, 0.0, translationVector)
    translationVector[0,0] *= 1
    translationVector[1,0] *= -1
    translationVector[2,0] *= -1
    w = cv.CreateMat(3,3,cv.CV_32FC1)
    ut = rotationMatrix = cv.CreateMat(3,3,cv.CV_32FC1)
    vt = rotationMatrix = cv.CreateMat(3,3,cv.CV_32FC1)
    cv.SVD(rotationMatrix,w,ut,vt,cv.CV_SVD_V_T + cv.CV_SVD_U_T)

    cv.GEMM(ut, vt, 1.0, None, 0.0, rotationMatrix)
    #print rotationMatrix

    c0 = cv.CreateMat(3,1,cv.CV_32FC1)
    c1 = cv.CreateMat(3,1,cv.CV_32FC1)
    c2 = cv.CreateMat(3,1,cv.CV_32FC1)
    c3 = cv.CreateMat(3,1,cv.CV_32FC1)
    c1[0,0] = 1
    c2[1,0] = 1
    c0[2,0] = 160
    c1[2,0] = 160
    c2[2,0] = 160
    c3[2,0] = 161

    p0 = cv.CreateMat(3,1,cv.CV_32FC1)
    p1 = cv.CreateMat(3,1,cv.CV_32FC1)
    p2 = cv.CreateMat(3,1,cv.CV_32FC1)
    p3 = cv.CreateMat(3,1,cv.CV_32FC1)

    cv.GEMM(cameraMatrix, c0, 1.0, None, 0.0, p0)
    cv.GEMM(cameraMatrix, c1, 1.0, None, 0.0, p1)
    cv.GEMM(cameraMatrix, c2, 1.0, None, 0.0, p2)
    cv.GEMM(cameraMatrix, c3, 1.0, None, 0.0, p3)
    print p0[0,0],p0[1,0]
    print p1[0,0],p1[1,0]
    print p2[0,0],p2[1,0]
    print p3[0,0],p3[1,0]
    p0 = (int(p0[0,0]),int(p0[1,0]))
    p1 = (int(p1[0,0]),int(p1[1,0]))
    p2 = (int(p2[0,0]),int(p2[1,0]))
    p3 = (int(p3[0,0]),int(p3[1,0]))

    t1 = cv.CreateMat(3,1,cv.CV_32FC1)
    t2 = cv.CreateMat(3,1,cv.CV_32FC1)
    t1[0,0] = 1.0
    t1[1,0] = 0.0
    t1[2,0] = 1.0
    cv.GEMM(inverseCameraMatrix,t1,1.0,None,0.0,t2) 
    cv.Line(im_in,p0,p1,(255,0,0))
    cv.Line(im_in,p0,p2,(255,0,0))
    cv.Line(im_in,p0,p3,(255,0,0))

    cv.ShowImage("Window", im_in)
    cv.WaitKey(0)
# use homogeneous coordinates
#p = numpy.array([point[0],point[1],1])
# convert the point from camera to display coordinates
#p = numpy.dot(matrix,p)
# normalize it
#point = (p[0]/p[2], p[1]/p[2])

    return

def sideBySideMatchView(gray1,gray2,hessianThresh):
    #p1,p2 = imageMatch(gray1,gray2,hessianThresh)
    p1,p2 = getSingleMatchFeatures(gray1, gray2, 1)
    w1, h1 = cv.GetSize(gray1)[:2]
    w2, h2 = cv.GetSize(gray2)[:2]
    vis = numpy.zeros((max(h1, h2), w1+w2), numpy.uint8)
    vis[:h1, :w1] = gray1
    vis[:h2, w1:w1+w2] = gray2

    vis = cv.fromarray(vis)
    visc = cv.CreateMat(max(h1, h2), w1+w2, cv.CV_8UC3)
    cv.CvtColor(vis, visc, cv.CV_GRAY2BGR)

    # Connect matching features with a line
    for i in range(len(p1)):
	x1 = int(p1[i][0][0])
	y1 = int(p1[i][0][1])
	x2 = int(p2[i][0][0])
	y2 = int(p2[i][0][1])
	randColor = (numpy.random.randint(0,255),numpy.random.randint(0,255),numpy.random.randint(0,255))
	cv.Line(visc, (x1,y1), (x2+w1, y2), randColor)

    # Add feature markers
    visc = overlayKeyPoints(visc,p1,(255,0,255))
    visc = overlayKeyPoints(visc,p2,(255,0,255),(w1,0))

    # Add info
    info = "Stable Matches: " + str(len(p1))
    cv.PutText(visc, info, (5,h2-10), FONT, (255,0,0))
    return visc

def overlayMatchView(gray1,gray2,hessianThresh):
    #p1,p2 = imageMatch(gray1,gray2,hessianThresh)
    p1,p2 = getSingleMatchFeatures(gray1, gray2, 1)
    w1, h1 = cv.GetSize(gray1)[:2]
    w2, h2 = cv.GetSize(gray2)[:2]

    visc = gray2

    # Add feature markers
    visc = overlayKeyPoints(visc,p1,(0,0,255))
    visc = overlayKeyPoints(visc,p2,(0,255,0))
    # Connect matching features with a line
    for i in range(len(p1)):
	x1 = int(p1[i][0][0])
	y1 = int(p1[i][0][1])
	x2 = int(p2[i][0][0])
	y2 = int(p2[i][0][1])

	cv.Line(visc, (x1,y1), (x2, y2), (255,0,0))


    # Add info
    info = "Stable Matches: " + str(len(p1))
    cv.PutText(visc, info, (5,h2-10), FONT, (255,0,0))

    return visc

def surfMatchVideoDemo(template = BLACK_CUBE,thresh=0.5):

    templateGray = cv.LoadImageM("black_cube.png", cv.CV_LOAD_IMAGE_GRAYSCALE)

    # capture from webcam and match surf features seen in video with example image
    cv.NamedWindow("Matchings", cv.CV_WINDOW_AUTOSIZE)
    camera_index = 0
    capture = cv.CaptureFromCAM(camera_index)
    frame = cv.QueryFrame(capture)

    w1, h1 = cv.GetSize(templateGray)[:2]
    w2, h2 = cv.GetSize(frame)[:2]

    f1 = cv.CreateMat(frame.height, frame.width, cv.CV_8UC1)
    f2 = cv.CreateMat(frame.height, frame.width, cv.CV_8UC1)   

    cv.CvtColor(frame, f1, cv.CV_BGR2GRAY)

    key = cv.WaitKey(1)
    while key != 'q':
        frame = cv.QueryFrame(capture)
	cv.CvtColor(frame, f2, cv.CV_BGR2GRAY)

	# Find stable SURF features
	stableKeypoints,stableDescriptors = getStableFeatures(f1, f2)
	# Match against template's features
        (templateKeypoints, templateDescriptors) = getFeatures(templateGray)
        kdtree = KDTree(stableDescriptors)
        p1 = []
        p2 = []
	isMatched1 = [0]*len(templateKeypoints)
	isMatched2 = [0]*len(stableKeypoints)
	possibleMatchList = numpy.array([[0,0,0]])
        for i in range(len(templateDescriptors)):
	    (dist,index) = kdtree.query(templateDescriptors[i],len(stableKeypoints))

    	    sourceID = numpy.array([i]*len(dist))
	    subMatchList = numpy.array([sourceID,index,dist])
	    subMatchList = subMatchList.transpose()

	    possibleMatchList = numpy.append(possibleMatchList,subMatchList,axis=0)

#	sortedMatchList = sorted(possibleMatchList, key=lambda matchPair: matchPair[1])
	sortedMatchList = possibleMatchList[possibleMatchList[:,2].argsort()]

	for i in range(1,len(sortedMatchList)):
	    sourceIndex = int(sortedMatchList[i,0])
	    targetIndex = int(sortedMatchList[i,1])
	    dist = sortedMatchList[i,2]
	    if (isMatched1[sourceIndex] == 0)&(isMatched2[targetIndex] == 0)&(dist<thresh):
		isMatched1[sourceIndex] = 1
		isMatched2[targetIndex] = 1
		p1.append(templateKeypoints[sourceIndex])
		p2.append(stableKeypoints[targetIndex])

	# Set up Side by Side Display
        vis = numpy.zeros((max(h1, h2), w1+w2), numpy.uint8)
	vis[:h1, :w1] = templateGray
	vis[:h2, w1:w1+w2] = f2

	vis = cv.fromarray(vis)
	visc = cv.CreateMat(max(h1, h2), w1+w2, cv.CV_8UC3)
	cv.CvtColor(vis, visc, cv.CV_GRAY2BGR)

	# Connect matching features with a line
	for i in range(len(p1)):
	    x1 = int(p1[i][0][0])
	    y1 = int(p1[i][0][1])
	    x2 = int(p2[i][0][0])
	    y2 = int(p2[i][0][1])
	    cv.Line(visc, (x1,y1), (x2+w1, y2), (255,0,0))

	# Add feature markers
	visc = overlayKeyPoints(visc,p1,(255,0,255))
	visc = overlayKeyPoints(visc,p2,(255,0,255),(w1,0))

	
	cv.ShowImage("Matchings", visc)
	f1 = f2
	cv.WaitKey()
        # exit on q
        key = cv.WaitKey(1)


def surfStableFeatureVideoDemo(displayType=0):
    # capture from webcam and match surf features seen in video with example image
    cv.NamedWindow("w1", cv.CV_WINDOW_AUTOSIZE)
    camera_index = 0
    capture = cv.CaptureFromCAM(camera_index)
    frame = cv.QueryFrame(capture)

    f1 = cv.CreateMat(frame.height, frame.width, cv.CV_8UC1)
    f2 = cv.CreateMat(frame.height, frame.width, cv.CV_8UC1)   

    cv.CvtColor(frame, f1, cv.CV_BGR2GRAY)

    key = cv.WaitKey(1)
    while key != 'q':
        frame = cv.QueryFrame(capture)
	imSize = cv.GetSize(frame)
	
	cv.CvtColor(frame, f2, cv.CV_BGR2GRAY)
	if (displayType == 0):
	    sideBySide = sideBySideMatchView(f1,f2,1000)
            cv.ShowImage("w1", sideBySide)
	elif (displayType == 1):
	    stable = overlayMatchView(f1,f2,1000)
            cv.ShowImage("w1", stable)
	cv.WaitKey()

	f1 = f2

        # exit on q
        key = cv.WaitKey(1)
	#print key

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
	cv.Line(overlaid, (px, py), (int(px+r*numpy.sin(fdir/numpy.pi)), int(py+r*numpy.cos(fdir/numpy.pi))), color)
    return overlaid

#def 

def sortTest():
	test = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
	test2 = sorted(test, key=lambda student: student[2])

	print test2
