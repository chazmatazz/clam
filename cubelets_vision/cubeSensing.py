#!/usr/bin/env python
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
CUBE_ARRAY = "images/cubeArray.png"
IMAGE_WINDOW_NAME = "Image"
WEBCAM_WINDOW_NAME = "WebCam"
CENTER_IMAGE = "Center"
CONVOLVED_CENTER_IMAGE = "Convolved Center"
HASH_CIRCLES = "Hash Circles"
CAMERA_INDEX = 0
MATCH_SIZE = 5
BRIGHTENED_IMAGE = "Brightened Image"

def loadImage(path, type=cv.CV_LOAD_IMAGE_GRAYSCALE):
    """ load an image """
    return cv.LoadImageM(path, type)

def getFeatures(img, hessThresh=1000):
    """ get features of image """
    return cv.ExtractSURF(img, None, cv.CreateMemStorage(), (0, hessThresh, 3, 1))

def getMatchedFeatures(template, test, descriptor_radius):
    """ match features by distance
    return list of features (templateFeatureIndex, testFeatureIndices) """
    (templateKeypoints, templateDescriptors) = template
    (testKeypoints, testDescriptors) = test
    kdtree = KDTree(testDescriptors)
    results = []
    for i in range(len(templateDescriptors)):
        results += [(i, kdtree.query_ball_point(templateDescriptors[i], descriptor_radius))]
    return results

def rotationMatrix2x2(angle):
    sina = math.sin(angle)
    cosa = math.cos(angle)
    return numpy.matrix([[cosa, sina], [-sina, cosa]])

def findMatches(imSize, testImage, template, test, descriptor_radius=0.65, center_radius=20, center_threshold=0.03, exclusion_radius=75):
    """ match features to a template """
    # TODO annotate matches with features
    (width, height) = imSize
    matchedFeatures = getMatchedFeatures(template, test, descriptor_radius)
    
    (templateKeypoints, templateDescriptors) = template
    (testKeypoints, testDescriptors) = test
    
    # retrieve centers in image space
    centers = []
    for (templateFeatureIndex, testFeatureIndices) in matchedFeatures:
        (templatePoint, templateLaplacian, templateSize, templateDirection, templateHessian) = templateKeypoints[templateFeatureIndex]
        for testFeatureIndex in testFeatureIndices:
            (testPoint, testLaplacian, testSize, testDirection, testHessian) = testKeypoints[testFeatureIndex]
            # vote for center
            rot = rotationMatrix2x2(testDirection - templateDirection)
            (x,y) = templatePoint
            offset = (x * rot[0,0] + y * rot[1,0], x * rot[0,1] + y * rot[1,1])
            featureCenter = numpy.array(testPoint) - 1.0 * testSize/templateSize * numpy.array(offset)
            centers += [(testKeypoints[testFeatureIndex], featureCenter)]
    
    def zero(img):
        for i in range(height):
            for j in range(width):
                img[i,j] = 0

    # create a center image
    centerImage = cv.CreateMat(height, width, cv.CV_32FC1)
    zero(centerImage)

    for center in centers:
        (k, p) = center
        x = round(p[0])
        y = round(p[1])
        if 0 <= x and x < width and 0 <= y and y < height:
            centerImage[y, x] += 1
    
    cv.ShowImage(CENTER_IMAGE, centerImage)
    
    # find bright points in centers image
    convolvedCenterImage = cv.CreateMat(height, width, cv.CV_32FC1)
    cv.Smooth(centerImage, convolvedCenterImage, cv.CV_GAUSSIAN, center_radius*2+1, -1)
    
    cv.ShowImage(CONVOLVED_CENTER_IMAGE, convolvedCenterImage)
    
    # get average value of convolved center image
    
    sum = 0
    for i in range(height):
        for j in range(width):
            sum += convolvedCenterImage[i,j]
    
    
    brightenedImage = cv.CreateMat(height, width, cv.CV_32FC1)        
    for i in range(height):
        for j in range(width):
            brightenedImage[i, j] = convolvedCenterImage[i, j] * testImage[i, j] * sum / (height * width) 
            
    cv.ShowImage(BRIGHTENED_IMAGE, brightenedImage)
    
    # find points in centers image above threshold
    aboveThreshold = []
    for i in range(height):
        for j in range(width):
            if convolvedCenterImage[i,j] > center_threshold:
                aboveThreshold += [(convolvedCenterImage[i,j], (j,i))]

    aboveThresholdSorted = sorted(aboveThreshold, key=lambda elt: elt[0])

    # find spaced points in centers image
    hashImage = cv.CreateMat(height, width, cv.CV_8UC1) # 0 is candidate, not candidate otherwise
    zero(hashImage)
    
    matches = []
    for elt in aboveThresholdSorted:
        p = elt[1]
        (x,y) = p
        if hashImage[y, x] == 0:
            matches += [p]
            cv.Circle(hashImage, p, exclusion_radius, 255, -1)
    cv.ShowImage(HASH_CIRCLES, hashImage)
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

if __name__ == '__main__':
    try:
        matchTemplateImage()
    except rospy.ROSInterruptException: 
        pass
