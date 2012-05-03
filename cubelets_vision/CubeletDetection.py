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


def rotationMatrix2x2(radians):
    sina = math.sin(radians)
    cosa = math.cos(radians)
    return numpy.matrix([[cosa, sina], [-sina, cosa]])


    
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
		square = cv.CreateMat(1 + 2*radius, 1 + 2*radius, cv.CV_8UC1)
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
	# at each pixel, take the difference of values between the pixel above and pixel below,
	# and multiply it with the difference for the pixels to the left and right.
	filtered = cv.CreateMat(gray.height, gray.width, cv.CV_8UC1)

	margin = 2
	for y in range(margin/2,gray.height-margin):
		for x in range(margin/2,gray.width-margin):
			dy = abs(gray[y+1,x] - gray[y-1,x])
			dx = abs(gray[y,x-1] - gray[y,x+1])

			filtered[y,x] = dy * dx

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
	# To be used on an image that has had vhDeltaProductFilter applied
	filtered = cv.CreateMat(gray.height, gray.width, cv.CV_8UC1)
	cv.Copy(gray,filtered)
	#cv.ShowImage('preErode',filtered)

	# Initial Smooth, Erosion, and Threshold
	cv.Smooth(filtered, filtered, cv.CV_GAUSSIAN, 5, -1)
	for y in range(0,filtered.height):
		for x in range(0,filtered.width):
			if (filtered[y,x] < 255*initialThresh):
				filtered[y,x] = 0
			else:
				filtered[y,x] = 255
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
	cv.Erode(filtered, filtered, None, 3 );
	for y in range(0,filtered.height):
		for x in range(0,filtered.width):
			if filtered[y,x] > 255.0*0.1:
				filtered[y,x] = 255
			else:
				filtered[y,x] = 0

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

        cv.Set(combinedResult,500000000)

        for color in self.templates:
            result = cv.CreateMat(rh, rw, cv.CV_32FC1)
            cv.MatchTemplate(testImage, self.templates[color], result, cv.CV_TM_SQDIFF_NORMED)

            for i in range(combinedResult.height):
                for j in range(combinedResult.width):	    
                    if result[i,j] < combinedResult[i,j]:
                        combinedResult[i,j] = result[i,j]

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
            likeliestColor = ""
            for c in self.templateColors:
                diff = colorDist(cubeColor,self.templateColors[c])
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
                # Handle situations with ~90 degrees and 0 degrees in the same list
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
    dst = vhDeltaProductFilter(gray)
    dst = poolAndIsolationFilter(dst)

    color_dst = cv.CreateImage(cv.GetSize(gray), 8, 3)
    storage = cv.CreateMemStorage(0)
    lines = 0
    cv.Canny(dst, dst, 50, 200, 3)

    cv.CvtColor(dst, color_dst, cv.CV_GRAY2BGR)

    lines = cv.HoughLines2(dst, storage, cv.CV_HOUGH_PROBABILISTIC, 1, math.pi / 180, 10, 15, 10)

    for line in lines:
        cv.Line(color_dst, line[0], line[1], cv.CV_RGB(255, 0, 0), 1, 8)

    cv.ShowImage("Source", gray)
    cv.ShowImage("Hough", color_dst)
    # Returns list of end point pairs
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




