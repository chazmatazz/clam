""" Library for Cubelet detection """

#import roslib; roslib.load_manifest('cubelets_vision')
import sys
#import rospy
import cv, cv2
import numpy
import math
from scipy.spatial import KDTree
import xml.etree.ElementTree as ET
#from std_msgs.msg import String
#from cv_bridge import CvBridge, CvBridgeError

def colorPath(desc):
    """ from a test image description, return a color image path """
    (folder, n) = desc
    return "images/webcam/%s/%s.jpg" % (folder, n)

def depthPath(desc):
    """ from a test image description, return a depth image path """
    (folder, n) = desc
    return "images/%s/%s_depth.png" % (folder, n)

FOLDER = "low-res-white"
NUM_IMAGES=18
TEST_IMAGES = [(FOLDER, n) for n in range(1, NUM_IMAGES+1)]
TRAINING_IMAGE = colorPath(TEST_IMAGES[1])
TEST_IMAGE = colorPath(TEST_IMAGES[1])
TRAINING_XML = "images/webcam/low-res-white/truth_values.xml"
TRAINING_IMAGE_NAME = "2.jpg"

CUBE_RADIUS = 20
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
RESULT_IMAGE = "Result Image"
COMBINED_RESULTS = "Combined Results"
TEMPLATE_STR = "Template"

BLACK = "black"
RED = "red"
CLEAR = "clear"
BLUETOOTH = "bluetooth"

THRESHOLDS = {BLACK:0.1, RED:0.1, CLEAR:0.1, BLUETOOTH:0.1}
EDGETHRESHOLD = .72
ROTATIONDEGREES = 15
    
FONT = cv.InitFont(cv.CV_FONT_HERSHEY_PLAIN, 1, 1, 0, 3, 8)

def loadDepthImage(path):
    """ load a depth image into a mm image """
    img = cv.LoadImageM(path)
    result = cv.CreateMat(img.height, img.width, cv.CV_32FC1)
    for i in range(img.height):
        for j in range(img.width):
            (red, green, blue) = img[i,j]
            result[i,j] = red * (1 << 8) + green
    return result

def loadImage(path, typ=cv.CV_LOAD_IMAGE_GRAYSCALE):
    """ load an image """
    return cv.LoadImageM(path, typ)

def filterFeaturesByDisc(features, cube, cube_radius):
    """ Filter features based on a disc """
    cx = float(cube.attrib["x"])
    cy = float(cube.attrib["y"])
    (keypoints, descriptors) = features
    filtered_keypoints = []
    filtered_descriptors = []
    for i in range(len(keypoints)):
        ((x, y), laplacian, size, direction, hessian) = keypoints[i]
        diff_x = x - cx
        diff_y = y - cy
        if diff_x*diff_x + diff_y * diff_y < cube_radius*cube_radius:
            # use relative to center position
            new_keypoint = ((diff_x, diff_y), laplacian, size, direction, hessian)
            filtered_keypoints += [new_keypoint]
            filtered_descriptors += [descriptors[i]]
    
    return (filtered_keypoints, filtered_descriptors)

def filterForDistinctFeatures(features, radius=0.2):
    """ filter for distinctive features """
    (keypoints, descriptors) = features
    kdtree = KDTree(descriptors)
    filteredKeypoints = []
    filteredDescriptors = []
    for i in range(len(descriptors)):
        if len(kdtree.query_ball_point(descriptors[i], radius)) <= 1:
            filteredKeypoints += [keypoints[i]]
            filteredDescriptors += [descriptors[i]]
    return (filteredKeypoints, filteredDescriptors)

def grayscaleize(img):
    """ Convert a color image to grayscale """
    imSize = cv.GetSize(img)
    (width, height) = imSize
    newImg = cv.CreateMat(height, width, cv.CV_8UC1)
    cv.CvtColor(img, newImg, cv.CV_BGR2GRAY)
    return newImg

class RGBGrayDetector:
    # would have been better to convert to HSV and do it that way!
    def __init__(self, min_gray=63, b_span=50, g_span=50, r_span=50):
        self.min_gray = min_gray
        self.b_span = b_span
        self.g_span = g_span
        self.r_span = r_span
        
    def is_gray(self, pix):
        (b, g, r) = pix
        avg = (b + g + r) / 3
        return (avg > self.min_gray and b > avg - self.b_span and 
                b < avg + self.b_span and g > avg - self.g_span and 
                g < avg + self.g_span and r > avg - self.r_span and 
                r < avg + self.r_span)
    
def dereflectImage(img, rgb_gray_detector, flat_gray=(255, 255, 255)):
    """ detect gray and replace it with a flat color """
    imSize = cv.GetSize(img)
    (width, height) = imSize
    newImg = cv.CreateMat(height, width, cv.CV_8UC3)
    for i in range(height):
        for j in range(width):
            if rgb_gray_detector.is_gray(img[i,j]):
                newImg[i, j] = flat_gray
            else:
                newImg[i, j] = img[i, j]
    return newImg

def thresholdImage(img, rgb_gray_detector):
    """ detect gray and represent as black, rest white """
    imSize = cv.GetSize(img)
    (width, height) = imSize
    newImg = cv.CreateMat(height, width, cv.CV_8UC1)
    for i in range(height):
        for j in range(width):
            if rgb_gray_detector.is_gray(img[i,j]):
                newImg[i, j] = 0
            else:
                newImg[i, j] = 255
    return newImg

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
        (dist, index) = kdtree.query(templateDescriptors[i], 2)
        if ((dist[0] / dist[1]) < thresh):
            k1 = templateKeypoints[i]
            k2 = testKeypoints[index[0]]
    
            m1.append(k1)
            m2.append(k2)
    return m1, m2

def getStableFeatures(frame1, frame2, thresh=0.75):
    # Find stable SURF features between two frames
    (f1Keypoints, f1Descriptors) = getFeatures(frame1)
    (f2Keypoints, f2Descriptors) = getFeatures(frame2)
    kdtree = KDTree(f2Descriptors)

    stableKeypoints = []
    stableDescriptors = []
    for i in range(len(f1Descriptors)):
        (dist, index) = kdtree.query(f1Descriptors[i], 2)
        if ((dist[0] / dist[1]) < thresh):
            stableKeypoints.append(f2Keypoints[index[0]])
            stableDescriptors.append(f2Descriptors[index[0]])
    return stableKeypoints, stableDescriptors

def rotationMatrix2x2(radians):
    sina = math.sin(radians)
    cosa = math.cos(radians)
    return numpy.matrix([[cosa, sina], [-sina, cosa]])

def getInliers(templateImagePath=TRAINING_IMAGE, testImagePath=TEST_IMAGE, descriptor_radius=0.1):
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
        points = []
        cv.Circle(myMatchedImage, match, INLIER_SIZE, 255, -1)
        for upper_left in upper_lefts:
            (i, k, p) = upper_left
            x = round(p[0])
            y = round(p[1])
            if 0 <= x and x < width and 0 <= y and y < height:
                if myMatchedImage[y, x] == 255:
                    currentInliers += [upper_left]
        for node in currentInliers:
            (index, keyPoint, upperLeft) = node
            #print node
            
            points += [(keyPoint, templateKeypoints[index])]
        inliers += [(match, points)]
        cv.Circle(myMatchedImage, match, INLIER_SIZE, 0, -1)

    for i in inliers:
        print i
    cv.WaitKey()
    return inliers # in form of (inlier_point, array of keypoints that hit point)

def getUpperLefts(matchedFeatures, template, test):
    """ given a set of matched features, return a list of (templateFeatureIndex, testKeypoint, upperleftPosition) """
    (templateKeypoints, templateDescriptors) = template
    (testKeypoints, testDescriptors) = test
    upper_lefts = []
    for (templateFeatureIndex, testFeatureIndices) in matchedFeatures:
        (templatePoint, templateLaplacian, templateSize, templateDirection, templateHessian) = templateKeypoints[templateFeatureIndex]
        for testFeatureIndex in testFeatureIndices:
            (testPoint, testLaplacian, testSize, testDirection, testHessian) = testKeypoints[testFeatureIndex]
                    # vote for upper left
            rot = rotationMatrix2x2(math.radians(testDirection - templateDirection))
            (x, y) = templatePoint
            offset = (x * rot[0, 0] + y * rot[1, 0], x * rot[0, 1] + y * rot[1, 1])
            featureUpperLeft = numpy.array(testPoint) - numpy.array(offset)# * 1.0 * testSize/templateSize
            upper_lefts += [(templateFeatureIndex, testKeypoints[testFeatureIndex], featureUpperLeft)]
    return upper_lefts

def zeroC1(img):
    """ Zero a grayscale image """
    imSize = cv.GetSize(img)
    (width, height) = imSize
    for i in range(height):
        for j in range(width):
            img[i, j] = 0

def findMatches(imSize, testImage, template, test, descriptor_radius=0.5, vote_radius=30, vote_threshold=0.005, exclusion_radius=20, match_size=MATCH_SIZE):
    """ match features to a template """
    # TODO annotate matches with features
    (width, height) = imSize
    matchedFeatures = getMultiMatchFeatures(template, test, descriptor_radius)
    
    (templateKeypoints, templateDescriptors) = template
    (testKeypoints, testDescriptors) = test
    
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
            n = (idx * 255 * 255 * 255) / self.length
            return ((n / (255 * 255)) % 255, (n / 255) % 255, n % 255)

    def drawVote(img, color, testKeypoint, vote):
        (featurePoint, laplacian, size, direction, hessian) = testKeypoint
        p1 = (int(featurePoint[0]), int(featurePoint[1]))
        radians = math.radians(direction)
        p2 = (int(featurePoint[0] + size * math.cos(radians)), int(featurePoint[1] + size * math.sin(radians)))
        p3 = (int(vote[0]), int(vote[1]))
        cv.Circle(img, p1, size, color)
        cv.Line(img, p1, p2, color)
        cv.Line(img, p1, p3, color)
        
    colorIndex = ColorIndex(len(template[0]))
    for upper_left in upper_lefts:
        (templateFeatureIndex, testKeypoint, featureUpperLeft) = upper_left
        drawVote(voteImage, colorIndex.getColor(templateFeatureIndex), testKeypoint, featureUpperLeft)
    
    # create a upper left image
    upperLeftImage = cv.CreateMat(height, width, cv.CV_32FC1)
    zeroC1(upperLeftImage)

    for upper_left in upper_lefts:
        (i, k, p) = upper_left
        x = round(p[0])
        y = round(p[1])
        if 0 <= x and x < width and 0 <= y and y < height:
            upperLeftImage[y, x] += 1
    
    # find bright points in upper left image
    convolvedUpperLeftImage = cv.CreateMat(height, width, cv.CV_32FC1)
    cv.Smooth(upperLeftImage, convolvedUpperLeftImage, cv.CV_GAUSSIAN, vote_radius * 2 + 1, -1)
    
    # get average value of convolved upper left image
    sum = 0
    for i in range(height):
        for j in range(width):
            sum += convolvedUpperLeftImage[i, j]
    
    brightenedImage = cv.CreateMat(height, width, cv.CV_32FC1)        
    for i in range(height):
        for j in range(width):
            brightenedImage[i, j] = convolvedUpperLeftImage[i, j] * testImage[i, j] * sum / (height * width) 
    
    # find points in convolved upper left image above threshold
    aboveThreshold = []
    for i in range(height):
        for j in range(width):
            if convolvedUpperLeftImage[i, j] > vote_threshold:
                aboveThreshold += [(convolvedUpperLeftImage[i, j], (j, i))]

    aboveThresholdSorted = sorted(aboveThreshold, key=lambda elt: elt[0])
    aboveThresholdSorted.reverse()

    # find spaced points in upper left image
    hashImage = cv.CreateMat(height, width, cv.CV_8UC1) # 0 is candidate, not candidate otherwise
    zeroC1(hashImage)
    
    matches = []
    for elt in aboveThresholdSorted:
        p = elt[1]
        (x, y) = p
        if hashImage[y, x] == 0:
            matches += [p]
            cv.Circle(hashImage, p, exclusion_radius, 255, -1)
    
    featureImage = cv.CloneMat(testImage)
    
    for match in matches:
        cv.Circle(featureImage, match, match_size, randColorTriplet(), -1)

    return (voteImage, upperLeftImage, convolvedUpperLeftImage, brightenedImage, hashImage, featureImage)    

def surfMatch(templateImagePath=TRAINING_IMAGE, testImagePath=TEST_IMAGE, 
                       training_xml=TRAINING_XML, training_image_name=TRAINING_IMAGE_NAME, cube_radius=CUBE_RADIUS):
    """ create templates, then match an image against them """
    templateImage = loadImage(templateImagePath)
    tree = ET.parse(TRAINING_XML)
    cubes = tree.findall("image[@name='" + TRAINING_IMAGE_NAME + "']/cube")
    templates = [(cube.attrib["color"], filterForDistinctFeatures(filterFeaturesByDisc(getFeatures(templateImage), cube, cube_radius))) for cube in cubes]
    
    testImage = loadImage(testImagePath)
    test = getFeatures(testImage)
    imSize = cv.GetSize(testImage)
    for (color, template) in templates:
        (voteImage, upperLeftImage, convolvedUpperLeftImage, brightenedImage, hashImage, featureImage) = findMatches(imSize, testImage, template, test)
        cv.ShowImage(VOTING_IMAGE + " " + color, voteImage)
        #cv.ShowImage(UPPER_LEFT_IMAGE + " " + color, upperLeftImage)
        #cv.ShowImage(CONVOLVED_UPPER_LEFT_IMAGE + " " + color, convolvedUpperLeftImage)
        #cv.ShowImage(BRIGHTENED_IMAGE + " " + color, brightenedImage)
        cv.ShowImage(HASH_CIRCLES + " " + color, hashImage)
        #cv.ShowImage(IMAGE_WINDOW_NAME + " " + color, featureImage)
    cv.WaitKey()
    
def templateMatch(testImages=TEST_IMAGES, templateImagePath=TRAINING_IMAGE, 
                       training_xml=TRAINING_XML, training_image_name=TRAINING_IMAGE_NAME, cube_radius=CUBE_RADIUS, thresholds=THRESHOLDS):
    """ match using template match """
    
    templateImage = cv.LoadImageM(templateImagePath)
    # cv.ShowImage(TRAINING_IMAGE, templateImage)
    
    tree = ET.parse(TRAINING_XML)
    cubes = tree.findall("image[@name='" + TRAINING_IMAGE_NAME + "']/cube")
    
    templates = {}
    for cube in cubes:
        color = cube.attrib["color"]
        x = int(cube.attrib["x"])
        y = int(cube.attrib["y"])
        template = cv.CreateMat(cube_radius*2, cube_radius*2, cv.CV_8UC3)
        for i in range(cube_radius*2):
            for j in range(cube_radius*2):
                template[i,j] = templateImage[y+i-cube_radius, x+j-cube_radius]
        #template = subImage(templateImage,(x,y),cube_radius)
        # cv.ShowImage("%s %s" % (TEMPLATE_STR, color), template)
        templates[color] = template
            
    for testImagePath in testImages:
        testImage = cv.LoadImageM(testImagePath)
        # cv.ShowImage("%s %s" % (testImagePath, TEST_IMAGE), testImage)
        results = {}
        rh = 1+testImage.height-cube_radius*2
        rw = 1+testImage.width-cube_radius*2
        for color in templates:
            result = cv.CreateMat(rh, rw, cv.CV_32FC1)
            cv.MatchTemplate(testImage, templates[color], result, cv.CV_TM_SQDIFF_NORMED)
            minval = 0
            results[color] = []
            while minval < thresholds[color]:
                # cv.ShowImage("%s %s %s" % (testImagePath, RESULT_IMAGE, color), result)
                (minval, maxval, (min_x, min_y), maxloc) = cv.MinMaxLoc(result)
                print minval
                results[color] += [(min_x + cube_radius, min_y + cube_radius)]
                cv.Circle(result, (min_x, min_y), cube_radius*2, maxval, -1)
        
        for color in results:
            for p in results[color]:
                pass
                
        combinedResults = cv.CloneMat(testImage)
                
        for color in results:
            for p in results[color]:
                    cv.Circle(combinedResults, p, cube_radius, getColor(color))
        cv.ShowImage("%s %s" % (testImagePath, COMBINED_RESULTS), combinedResults)
    cv.WaitKey()
    
def matchTemplateWebcam(templateImagePath=TRAINING_IMAGE, window_name=WEBCAM_WINDOW_NAME, camera_index=CAMERA_INDEX):
    """ create a template, then match webcam stream against it """
    cv.NamedWindow(window_name, cv.CV_WINDOW_AUTOSIZE)
    capture = cv.CaptureFromCAM(camera_index)
    
    template = getFeatures(loadImage(templateImagePath))

    #TODO

def randColorTriplet():
    """ create a random color """
    return (numpy.random.randint(0, 255), numpy.random.randint(0, 255), numpy.random.randint(0, 255))



def sideBySideMatchView(gray1, gray2, hessianThresh):
    #p1,p2 = imageMatch(gray1,gray2,hessianThresh)
    p1, p2 = getSingleMatchFeatures(gray1, gray2, 1)
    w1, h1 = cv.GetSize(gray1)[:2]
    w2, h2 = cv.GetSize(gray2)[:2]
    vis = numpy.zeros((max(h1, h2), w1 + w2), numpy.uint8)
    vis[:h1, :w1] = gray1
    vis[:h2, w1:w1 + w2] = gray2

    vis = cv.fromarray(vis)
    visc = cv.CreateMat(max(h1, h2), w1 + w2, cv.CV_8UC3)
    cv.CvtColor(vis, visc, cv.CV_GRAY2BGR)

    # Connect matching features with a line
    for i in range(len(p1)):
        x1 = int(p1[i][0][0])
        y1 = int(p1[i][0][1])
        x2 = int(p2[i][0][0])
        y2 = int(p2[i][0][1])
        randColor = (numpy.random.randint(0, 255), numpy.random.randint(0, 255), numpy.random.randint(0, 255))
        cv.Line(visc, (x1, y1), (x2 + w1, y2), randColor)

    # Add feature markers
    visc = overlayKeyPoints(visc, p1, (255, 0, 255))
    visc = overlayKeyPoints(visc, p2, (255, 0, 255), (w1, 0))

    # Add info
    info = "Stable Matches: " + str(len(p1))
    cv.PutText(visc, info, (5, h2 - 10), FONT, (255, 0, 0))
    return visc

def overlayMatchView(gray1, gray2, hessianThresh):
    #p1,p2 = imageMatch(gray1,gray2,hessianThresh)
    p1, p2 = getSingleMatchFeatures(gray1, gray2, 1)
    w1, h1 = cv.GetSize(gray1)[:2]
    w2, h2 = cv.GetSize(gray2)[:2]

    visc = gray2

    # Add feature markers
    visc = overlayKeyPoints(visc, p1, (0, 0, 255))
    visc = overlayKeyPoints(visc, p2, (0, 255, 0))
    # Connect matching features with a line
    for i in range(len(p1)):
        x1 = int(p1[i][0][0])
        y1 = int(p1[i][0][1])
        x2 = int(p2[i][0][0])
        y2 = int(p2[i][0][1])
        cv.Line(visc, (x1, y1), (x2, y2), (255, 0, 0))


    # Add info
    info = "Stable Matches: " + str(len(p1))
    cv.PutText(visc, info, (5, h2 - 10), FONT, (255, 0, 0))

    return visc

def surfMatchVideoDemo(template=TRAINING_IMAGE, thresh=0.5):

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
        stableKeypoints, stableDescriptors = getStableFeatures(f1, f2)
        # Match against template's features
        (templateKeypoints, templateDescriptors) = getFeatures(templateGray)
        kdtree = KDTree(stableDescriptors)
        p1 = []
        p2 = []
        isMatched1 = [0] * len(templateKeypoints)
        isMatched2 = [0] * len(stableKeypoints)
        possibleMatchList = numpy.array([[0, 0, 0]])
        for i in range(len(templateDescriptors)):
            (dist, index) = kdtree.query(templateDescriptors[i], len(stableKeypoints))
            sourceID = numpy.array([i] * len(dist))
            subMatchList = numpy.array([sourceID, index, dist])
            subMatchList = subMatchList.transpose()
    
            possibleMatchList = numpy.append(possibleMatchList, subMatchList, axis=0)
    
        # sortedMatchList = sorted(possibleMatchList, key=lambda matchPair: matchPair[1])
        sortedMatchList = possibleMatchList[possibleMatchList[:, 2].argsort()]
    
        for i in range(1, len(sortedMatchList)):
            sourceIndex = int(sortedMatchList[i, 0])
            targetIndex = int(sortedMatchList[i, 1])
            dist = sortedMatchList[i, 2]
            if (isMatched1[sourceIndex] == 0) & (isMatched2[targetIndex] == 0) & (dist < thresh):
                isMatched1[sourceIndex] = 1
                isMatched2[targetIndex] = 1
                p1.append(templateKeypoints[sourceIndex])
                p2.append(stableKeypoints[targetIndex])
    
        # Set up Side by Side Display
            vis = numpy.zeros((max(h1, h2), w1 + w2), numpy.uint8)
        vis[:h1, :w1] = templateGray
        vis[:h2, w1:w1 + w2] = f2
    
        vis = cv.fromarray(vis)
        visc = cv.CreateMat(max(h1, h2), w1 + w2, cv.CV_8UC3)
        cv.CvtColor(vis, visc, cv.CV_GRAY2BGR)
    
        # Connect matching features with a line
        for i in range(len(p1)):
            x1 = int(p1[i][0][0])
            y1 = int(p1[i][0][1])
            x2 = int(p2[i][0][0])
            y2 = int(p2[i][0][1])
            cv.Line(visc, (x1, y1), (x2 + w1, y2), (255, 0, 0))
    
        # Add feature markers
        visc = overlayKeyPoints(visc, p1, (255, 0, 255))
        visc = overlayKeyPoints(visc, p2, (255, 0, 255), (w1, 0))
    
        
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
            sideBySide = sideBySideMatchView(f1, f2, 1000)
            cv.ShowImage("w1", sideBySide)
        elif (displayType == 1):
            stable = overlayMatchView(f1, f2, 1000)
            cv.ShowImage("w1", stable)
        cv.WaitKey()

        f1 = f2

        # exit on q
        key = cv.WaitKey(1)

def overlayKeyPoints(imgMat, keyPoints, color, offset=(0, 0)):
    # Overlay a set of surf feature marker onto an image
    # If grayscale, convert to color
    if isinstance(imgMat[0, 0], float):
        imSize = cv.GetSize(imgMat)
        overlaid = cv.CreateMat(imSize[1], imSize[0], cv.CV_8UC3)
        cv.CvtColor(imgMat, overlaid, cv.CV_GRAY2BGR)
    else:
        overlaid = imgMat
    
    for ((x, y), laplacian, size, fdir, hessian) in keyPoints:
        r = int(0.5 * size)
        px = int(x + offset[0])
        py = int(y + offset[1])
        cv.Circle(overlaid, (px, py), r, color)
        #cv.Line(overlaid, (px, py), (int(px + r * numpy.sin(fdir / numpy.pi)), int(py + r * numpy.cos(fdir / numpy.pi))), color)
    return overlaid

def rotateImage(img,center,degrees):
    # Rotate an image within its frame bounds 
    # (width and height do not change)
    if (img.channels == 3):
        rotated = cv.CreateMat(img.height, img.width, cv.CV_8UC3)
    elif (img.channels == 1):
        rotated = cv.CreateMat(img.height, img.width, cv.CV_8UC1)

    mapMatrix = cv.CreateMat(2, 3, cv.CV_32FC1)
    cv.GetRotationMatrix2D(center, degrees, 1, mapMatrix)
    cv.WarpAffine(img, rotated, mapMatrix)

    return rotated

def subImage(img,center,radius):
    # Return a 2r x 2r square of the source image
    (x,y) = center
    if (img.channels == 3):
        square = cv.CreateMat(1 + 2*radius, 1 + 2*radius, cv.CV_8UC3)
    elif (img.channels == 1):
        square = cv.CreateMat(1 + 2*radius, 1 + 2*radius, cv.CV_8UFC1)
    else:
        # No reason for us to have any other channel count
        return -1
    cx = radius + 1
    cy = radius + 1
    for i in range(0,2*radius+1):
        for j in range(0,2*radius+1):
            square[i,j] = img[y+(i-radius), x+(j-radius)]
    return square


def vhDeltaProductFilter(gray):
    """ 
    at each pixel, take the difference of values between the pixel above and pixel below,
    and multiply it with the difference for the pixels to the left and right.
    """
    edge_kernel = numpy.array([-1.0, 0, 1.0], numpy.float32)
    id_kernel = numpy.array([0, 1.0, 0], numpy.float32)
    
    numpy_gray = numpy.array(gray, numpy.float32)
    
    horiz = numpy.zeros((gray.height, gray.width), numpy.float32)
    vert = numpy.zeros((gray.height, gray.width), numpy.float32)
    filtered = cv.CreateMat(gray.height, gray.width, cv.CV_8UC1)
    
    horiz = cv2.sepFilter2D(numpy_gray, -1, id_kernel, edge_kernel)
    vert = cv2.sepFilter2D(numpy_gray, -1, edge_kernel, id_kernel)
    horiz = numpy.abs(horiz)
    vert = numpy.abs(vert)
    cv.Mul(cv.fromarray(horiz), cv.fromarray(vert), filtered)

    return filtered

def vhDeltaProductFilter2(gray):
    # Take Difference from each direction seperately, and scale down
    filtered = cv.CreateMat(gray.height, gray.width, cv.CV_8UC1)

    margin = 2
    for y in range(margin/2,gray.height-margin):
        for x in range(margin/2,gray.width-margin):
            dy1 = abs(gray[y+1,x] - gray[y,x])
            dy2 = abs(gray[y-1,x] - gray[y,x])
            dx1 = abs(gray[y,x+1] - gray[y,x])
            dx2 = abs(gray[y,x-1] - gray[y,x])

            filtered[y,x] = dy1 * dy2 * dx1 * dx2 / 255

    return filtered

def poolAndIsolationFilter(gray,initialThresh=0.15,iterations=10):
    filtered = cv.CreateMat(gray.height, gray.width, cv.CV_8UC1)
    cv.Copy(gray,filtered)
    #cv.ShowImage('preErode',filtered)

    # Initial Smooth, Erosion, and Threshold
    cv.Smooth(filtered, filtered, cv.CV_GAUSSIAN, 5, -1)

    cv.Threshold(filtered, filtered, 255*initialThresh, 255, cv.CV_THRESH_BINARY)
    
    
    cv.Erode(filtered, filtered, None, 1 );
    #cv.ShowImage('postErode',filtered)

    # Repeatedly Dilate and Erode to Merge Nearby Bright Spots
    for i in range(0,iterations):
        cv.Smooth(filtered, filtered, cv.CV_GAUSSIAN, 5, -1)
        cv.Dilate(filtered, filtered, None, 1 )
        cv.Erode(filtered, filtered, None, 1 )

    #cv.ShowImage('postErosionDilation',filtered)

    # Smooth, Erode and Threshold a final time
    cv.Smooth(filtered, filtered, cv.CV_GAUSSIAN, 5, -1)
    cv.Erode(filtered, filtered, None, 3 )
    
    cv.Threshold(filtered, filtered, 255*0.1, 255, cv.CV_THRESH_BINARY)
    
    return filtered

def avgPointCenter(matchList):
    avgx = 0
    avgy = 0
    for match in matchList:
        avgx += match[1][0]
        avgy += match[1][1]

    avgx = avgx / len(matchList)
    avgy = avgy / len(matchList)

    return (avgx,avgy)

def getTemplateInfo():
    templateInfoPath = "images/webcam/low-res-white/template_info.xml"
    tree = ET.parse(templateInfoPath)
    MASTER_TEMPLATE_NAME = "isolatedTemplates.png"
    cubes = tree.findall("image[@name='" + MASTER_TEMPLATE_NAME + "']/cube")
    cubeList = {}

    for cube in cubes:
        color = cube.attrib["color"]
        x = int(cube.attrib["x"])
        y = int(cube.attrib["y"])
        r = int(cube.attrib["r"])
        if (color in cubeList):
            cubeList[color].append(((x,y),r))
        else:
            cubeList[color] = [((x,y),r)]

    return cubeList

def rotationMatchDemo(angles = 6):
    # Check in 15 degree increments
    targetPath = "images/webcam/low-res-white/18.jpg"
    templatePath = "images/webcam/low-res-white/isolatedTemplates.png"
    target = cv.LoadImageM(targetPath, cv.CV_LOAD_IMAGE_COLOR)
    templateSet = cv.LoadImageM(templatePath, cv.CV_LOAD_IMAGE_COLOR)

    markerColor = {'black':(0,0,0),'red':(0,0,255),'clear':(255,255,255),'bluetooth':(255,0,0)}

    # Test over a set of example images
    cubeList = getTemplateInfo()
    for cubeType in cubeList:
        print "Cube Type: ",cubeType
        possibleMatches = []
        for (templateCenter,templateRadius) in cubeList[cubeType]:
#    for templateCenter in templateCenters:
            # For each example image, test several rotations
            for i in range(0,angles):
                # Create Rotated Template
                degrees = i*(90/angles)
                rotated = rotateImage(templateSet,templateCenter,degrees)
                #cv.ShowImage('templater'+str(i),rotated)
                template = subImage(rotated,templateCenter,templateRadius)
                #cv.ShowImage('template'+str(i),template)

                # Establish Offset Parameters
                rh = 1+target.height-template.height
                rw = 1+target.width-template.width
                offsetX = target.width-rw
                offsetY = target.height-rh

                # Perform Matching
                result = cv.CreateMat(rh, rw, cv.CV_32FC1)
                cv.MatchTemplate(target, template, result, cv.CV_TM_SQDIFF_NORMED)

                # Locate Points of Best Match
                resultList = []
                for y in range(0,result.height):
                    for x in range(0,result.width):
                        resultList.append((result[y,x],(x,y)))

                sortedResults = sorted(resultList, key=lambda point: point[0])
                for j in range(0,10):
                    #print sortedResults[j], sortedResults[0][0]/sortedResults[j][0]
                    # Collect and mark possible matches
                    # Consider a possible match it is within 90% of the best match
                    # NOTE: Need to add threshold on best match to determine if
                    # there are no matches in the image.
                    if (sortedResults[0][0]/sortedResults[j][0] > 0.9):
                        cv.Circle(result, sortedResults[j][1], 3, 255)
                        possibleMatches.append(sortedResults[j])
                # Show Match level for each rotation
                #cv.ShowImage("result"+str(i),result)
            #
        # Cluster based on distance
        clusters = [[]]
        cubeRadius = 15
        for match in possibleMatches:
            k = 0
            foundCluster = 0
            while (k < len(clusters))&(foundCluster == 0):
                if (len(clusters[k]) == 0):
                    clusters[k].append(match)
                    foundCluster = 1
                else:
                    clusterCenter = avgPointCenter(clusters[k])
                    if (pixelDist(clusterCenter,match[1]) < cubeRadius):
                        clusters[k].append(match)
                        foundCluster = 1
                k += 1
            if (foundCluster == 0):
                clusters.append([match])

        # Calculate the center of each cluster, 
        # offset to match the center of the template
        clusterCenters = []
        for cluster in clusters:
            center = avgPointCenter(cluster)
            center = (center[0] + templateRadius,center[1] + templateRadius)

            clusterCenters.append(center)

        # Mark the matches on the target image
        print "Marking Cube Type: ",cubeType
        print "Matches: ", len(clusterCenters)
        for center in clusterCenters:
            color = markerColor[cubeType]
            cv.Circle(target,center,templateRadius,color)


    cv.ShowImage("clusters",target)

    cv.WaitKey()

class EdgeTemplate:
    """ creates an edge template """
    def __init__(self, templateImagePath, training_xml, training_image_name, cube_radius, 
                 rotation_degrees, edge_threshold):
        """ create the template """
        self.cube_radius = cube_radius
        self.edge_threshold = edge_threshold
        
        templateImage = cv.LoadImageM(templateImagePath)
        templateGray = cv.CreateMat(templateImage.height, templateImage.width, cv.CV_8UC1)
        cv.CvtColor(templateImage, templateGray, cv.CV_BGR2GRAY)
        edgeImage = vhDeltaProductFilter(templateGray)
    
        tree = ET.parse(training_xml)
        cubes = tree.findall("image[@name='" + training_image_name + "']/cube")
        
        self.templates = {}
        self.templateColors = {}
        for cube in cubes:
            color = cube.attrib["color"]
            x = int(cube.attrib["x"])
            y = int(cube.attrib["y"])
            template = cv.CreateMat(cube_radius*2, cube_radius*2, cv.CV_8UC1)
            counter = 0
            # Save the average color of the template cube
            self.templateColors[color] = avgColor(templateImage,(x,y),cube_radius)
            # Take the sub image
            for i in range(cube_radius*2):
                for j in range(cube_radius*2):
                    template[i,j] = edgeImage[y+i-cube_radius, x+j-cube_radius]
            # Rotate the sub image and save it
            for multiplier in range(90/rotation_degrees):
                ry = template.height/2
                rx = template.width/2
                rotated = rotateImage(template,(ry,rx),multiplier*rotation_degrees)
    
                self.templates[color+str(counter)] = rotated
                counter += 1
    
    def matchTemplate(self, img):
        """ match an image to the template """
        grayTestImage = cv.CreateMat(img.height, img.width, cv.CV_8UC1)
        cv.CvtColor(img, grayTestImage, cv.CV_BGR2GRAY)

        testImage = vhDeltaProductFilter(grayTestImage)

        results = []
        rh = 1+testImage.height-self.cube_radius*2
        rw = 1+testImage.width-self.cube_radius*2

        combinedResult = cv.CreateMat(rh, rw, cv.CV_32FC1)

        first = True

        for color in self.templates:
            result = cv.CreateMat(rh, rw, cv.CV_32FC1)
            cv.MatchTemplate(testImage, self.templates[color], result, cv.CV_TM_SQDIFF_NORMED)
            if first:
                combinedResult = cv.CloneMat(result)
            else:
                cv.Min(result, combinedResult, combinedResult)

            first = False

        (minval, maxval, (min_x, min_y), maxloc) = cv.MinMaxLoc(combinedResult)

        while minval < self.edge_threshold:                    
            results += [(min_x + self.cube_radius, min_y + self.cube_radius)]
            # draw the exclusion circle
            cv.Circle(combinedResult, (min_x, min_y), self.cube_radius*2, maxval, -1)
            (minval, maxval, (min_x, min_y), maxloc) = cv.MinMaxLoc(combinedResult)

        # Find line segments in the original grayscale image
        lines = findLineSegments(grayTestImage)
        ret = []
        for p in results:
            (x,y) = p
            cubeColor = avgColor(img, p, self.cube_radius)
            # Match the color to the given template cubes
            minDist = ()
            color = ""
            for c in self.templateColors:
                diff = colorDist(cubeColor, self.templateColors[c])
                if diff < minDist:
                    minDist = diff
                    color = c
            # Find the rotation of the cube
            angle = findRotation(lines,p,self.cube_radius*1.5)
            ret += [(x, y, angle, color)]
        
        return ret
    
def edgeTemplateMatch(test_images=TEST_IMAGES, templateImagePath=TRAINING_IMAGE, training_xml=TRAINING_XML, 
                 training_image_name=TRAINING_IMAGE_NAME, cube_radius=CUBE_RADIUS, 
                 rotation_degrees=ROTATIONDEGREES, edge_threshold=EDGETHRESHOLD):
    """ match a set of images to a template """
    et = EdgeTemplate(templateImagePath, training_xml, training_image_name, cube_radius, rotation_degrees, edge_threshold)
    for desc in test_images:
        img = cv.LoadImageM(colorPath(desc))
        drawMatch(et, img, et.matchTemplate(img))
        (folder, n) = desc
        cv.ShowImage("%s %s %s" % (folder, n, COMBINED_RESULTS), img)
        # cv.SaveImage("images/report images/results/%s/%s.png" % desc, img)
    
    cv.WaitKey()

def getColor(color):
    if color == BLACK:
        return (0, 0, 0)
    elif color == RED:
        return (0, 0, 255)
    elif color == CLEAR:
        return (255,255,255)
    elif color == BLUETOOTH:
        return (255, 0, 0)
        
def drawMatch(edge_template, img, matches):
    """ create a match image """
    white = (255, 255, 255)
    dia = 6
    for match in matches:
        (x, y, angle, color) = match
        p = (x, y)
        cv.Circle(img, p, dia, getColor(color),-1)
        cv.Circle(img, p, dia, white)
        cv.Circle(img, p, edge_template.cube_radius, white)
        if angle:
            cv.Line(img, p, (int(x+edge_template.cube_radius*numpy.cos(angle)), int(y-edge_template.cube_radius*numpy.sin(angle))), white)            
                
def findRotation(lines,point,radius):
    """ Finds the avg (angle % pi/2) of lines within one radius of the point """
    angles = []
    for line in lines:
        isNearby = (pixelDist(point,line[0]) < 2*radius)|(pixelDist(point,line[1]) < 2*radius)
        if (isNearby)&(point2LineDist(point,line[0], line[1]) <= radius):
            if line[0][0] < line[1][0]:
                (x1,y1) = line[0]
                (x2,y2) = line[1]
            else:
                (x1,y1) = line[1]
                (x2,y2) = line[0]

            theta = math.atan2(y1-y2,x2-x1) % (math.pi / 2.0)

            if len(angles) > 0:
                currAvg = float(sum(angles)) / len(angles)
                if abs(currAvg - theta) > abs(currAvg - (theta - math.pi/2)):
                    theta = theta - math.pi/2
                elif abs(currAvg - theta) > abs(currAvg - (theta + math.pi/2)):
                    theta = theta + math.pi/2

            angles.append(theta)
    if len(angles) > 0:
        angle = float(sum(angles)) / len(angles)
    else:
        angle = None
    return angle

def findLineSegments(gray):
    #dst = cv.CreateImage(cv.GetSize(gray), 8, 1)

    dst = vhDeltaProductFilter(gray)
    dst = poolAndIsolationFilter(dst)

    color_dst = cv.CreateImage(cv.GetSize(gray), 8, 3)
    storage = cv.CreateMemStorage(0)
    lines = 0
    cv.Canny(dst, dst, 50, 200, 3)

    cv.CvtColor(dst, color_dst, cv.CV_GRAY2BGR)

    #lines = cv.HoughLines2(dst, storage, cv.CV_HOUGH_PROBABILISTIC, 1, math.pi / 180, 30, 25, 10)
    lines = cv.HoughLines2(dst, storage, cv.CV_HOUGH_PROBABILISTIC, 1, math.pi / 180, 10, 15, 10)

    for line in lines:
        cv.Line(color_dst, line[0], line[1], cv.CV_RGB(255, 0, 0), 1, 8)

    # cv.ShowImage("Source", gray)
    # cv.ShowImage("Hough", color_dst)

    return lines

def point2LineDist(p0,p1,p2):
    # Finds the distance between a point p0 and a line connecting p1 and p2
    (x0,y0) = p0
    (x1,y1) = p1
    (x2,y2) = p2
    top = abs((x2-x1)*(y1-y0)-(x1-x0)*(y2-y1))
    bot = math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))
    dist = top / bot
    return dist

def avgColor(img,center,radius):
    # Find the average color in a subsquare of a color image (8UC3)
    (x,y) = center
    rgb = [0,0,0]
    for i in range(radius*2):
        for j in range(radius*2):
            rgb[0] = rgb[0] + img[y+i-radius, x+j-radius][0]
            rgb[1] = rgb[1] + img[y+i-radius, x+j-radius][1]
            rgb[2] = rgb[2] + img[y+i-radius, x+j-radius][2]
    rgb[0] = rgb[0] / (4*radius*radius)
    rgb[1] = rgb[1] / (4*radius*radius)
    rgb[2] = rgb[2] / (4*radius*radius)

    return rgb

def colorDist(rgb1,rgb2):
    # Calculate the distance between two colors (also works for hsv)
    dr = rgb1[0] - rgb2[0]
    dg = rgb1[1] - rgb2[1]
    db = rgb1[2] - rgb2[2]

    dist = math.sqrt(dr*dr + dg*dg + db*db)
    return dist

def pixelDist(p1,p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    d = math.sqrt(dx*dx + dy*dy)
    return d
