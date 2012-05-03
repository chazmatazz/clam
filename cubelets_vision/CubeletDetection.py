""" Library for Cubelet detection """

import roslib
roslib.load_manifest('cubelets_vision')
import sys
import rospy
import cv, cv2
import numpy
import math
import xml.etree.ElementTree as ET
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import threading
import time
import os

BLACK = "black"
RED = "red"
CLEAR = "clear"
BLUETOOTH = "bluetooth"

TYPE = "low-res-webcam"
FOLDER = "white"
NUM_IMAGES = 18
TEST_IMAGES = [("%s %s %s" % (TYPE, FOLDER, n), "images/%s/%s/%s.jpg" % (TYPE, FOLDER, n), "images/%s/%s/%s_depth.png" % (TYPE, FOLDER, n), "results/%s/%s/%s.png" % (TYPE, FOLDER, n), n) for n in range(1, NUM_IMAGES + 1)]
TRAINING_IMAGE = TEST_IMAGES[1]
TRAINING_XML = "images/%s/%s/truth_values.xml" % (TYPE, FOLDER)

class EdgeTemplate:
    """ creates an edge template """
    
    def loadDepthImage(self, path):
        """ load a depth image into a mm image """
        img = cv.LoadImageM(path)
        result = cv.CreateMat(img.height, img.width, cv.CV_32FC1)
        for i in range(img.height):
            for j in range(img.width):
                (red, green, blue) = img[i, j]
                result[i, j] = red * (1 << 8) + green
        return result

    def vhDeltaProductFilter(self, gray):
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

    def avgColor(self, img, center, radius):
        """ Find the average color in a subsquare of a color image (8UC3) """
        (x, y) = center
        rgb = [0, 0, 0]
        for i in range(radius * 2):
            for j in range(radius * 2):
                rgb[0] = rgb[0] + img[y + i - radius, x + j - radius][0]
                rgb[1] = rgb[1] + img[y + i - radius, x + j - radius][1]
                rgb[2] = rgb[2] + img[y + i - radius, x + j - radius][2]
        rgb[0] = rgb[0] / (4 * radius * radius)
        rgb[1] = rgb[1] / (4 * radius * radius)
        rgb[2] = rgb[2] / (4 * radius * radius)
    
        return rgb
    
    def __init__(self, training_xml, training_image, edge_threshold=0.72, rotation_degrees=15):
        """ create the template """
        def rotateImage(img, center, degrees):
            """ Rotate an image within its frame bounds (width and height do not change) """
            if (img.channels == 3):
                rotated = cv.CreateMat(img.height, img.width, cv.CV_8UC3)
            elif (img.channels == 1):
                rotated = cv.CreateMat(img.height, img.width, cv.CV_8UC1)
        
            mapMatrix = cv.CreateMat(2, 3, cv.CV_32FC1)
            cv.GetRotationMatrix2D(center, degrees, 1, mapMatrix)
            cv.WarpAffine(img, rotated, mapMatrix)
        
            return rotated
        
        self.edge_threshold = edge_threshold
        self.rotation_degrees = rotation_degrees
        
        (window_name, color_image, depth_image, result_image, n) = training_image
        templateImage = cv.LoadImageM(color_image)
        templateGray = cv.CreateMat(templateImage.height, templateImage.width, cv.CV_8UC1)
        cv.CvtColor(templateImage, templateGray, cv.CV_BGR2GRAY)
        edgeImage = self.vhDeltaProductFilter(templateGray)
    
        tree = ET.parse(training_xml)
        self.cube_radius = int(tree.getroot().attrib["cube_radius"])
        cubes = tree.findall("image[@id='%d']/cube" % n)
        
        self.templates = {}
        self.templateColors = {}
        for cube in cubes:
            color = cube.attrib["color"]
            x = int(cube.attrib["x"])
            y = int(cube.attrib["y"])
            template = cv.CreateMat(self.cube_radius * 2, self.cube_radius * 2, cv.CV_8UC1)
            counter = 0
            # Save the average color of the template cube
            self.templateColors[color] = self.avgColor(templateImage, (x, y), self.cube_radius)
            # Take the sub image
            for i in range(self.cube_radius * 2):
                for j in range(self.cube_radius * 2):
                    template[i, j] = edgeImage[y + i - self.cube_radius, x + j - self.cube_radius]
            # Rotate the sub image and save it
            for multiplier in range(90 / rotation_degrees):
                ry = template.height / 2
                rx = template.width / 2
                rotated = rotateImage(template, (ry, rx), multiplier * rotation_degrees)
    
                self.templates[color + str(counter)] = rotated
                counter += 1
    
    def matchTemplate(self, img):
        """ match an image to the template """
        # defs
        def avgPointCenter(matchList):
            avgx = 0
            avgy = 0
            for match in matchList:
                avgx += match[1][0]
                avgy += match[1][1]
        
            avgx = avgx / len(matchList)
            avgy = avgy / len(matchList)
        
            return (avgx, avgy)

        def findRotation(lines, point, radius):
            """ Finds the avg (angle % pi/2) of lines within one radius of the point """
            angles = []
            for line in lines:
                isNearby = (pixelDist(point, line[0]) < 2 * radius) | (pixelDist(point, line[1]) < 2 * radius)
                if (isNearby) & (point2LineDist(point, line[0], line[1]) <= radius):
                    if line[0][0] < line[1][0]:
                        (x1, y1) = line[0]
                        (x2, y2) = line[1]
                    else:
                        (x1, y1) = line[1]
                        (x2, y2) = line[0]
        
                    theta = math.atan2(y1 - y2, x2 - x1) % (math.pi / 2.0)
        
                    if len(angles) > 0:
                        currAvg = float(sum(angles)) / len(angles)
                        if abs(currAvg - theta) > abs(currAvg - (theta - math.pi / 2)):
                            theta = theta - math.pi / 2
                        elif abs(currAvg - theta) > abs(currAvg - (theta + math.pi / 2)):
                            theta = theta + math.pi / 2
        
                    angles.append(theta)
            if len(angles) > 0:
                angle = float(sum(angles)) / len(angles)
            else:
                angle = None
            return angle
        
        def findLineSegments(gray):
            dst = self.vhDeltaProductFilter(gray)
            dst = poolAndIsolationFilter(dst)
        
            color_dst = cv.CreateImage(cv.GetSize(gray), 8, 3)
            storage = cv.CreateMemStorage(0)
            lines = 0
            cv.Canny(dst, dst, 50, 200, 3)
        
            cv.CvtColor(dst, color_dst, cv.CV_GRAY2BGR)
        
            lines = cv.HoughLines2(dst, storage, cv.CV_HOUGH_PROBABILISTIC, 1, math.pi / 180, 10, 15, 10)
        
            for line in lines:
                cv.Line(color_dst, line[0], line[1], cv.CV_RGB(255, 0, 0), 1, 8)
        
            return lines
        
        def point2LineDist(p0, p1, p2):
            """ Finds the distance between a point p0 and a line connecting p1 and p2 """
            (x0, y0) = p0
            (x1, y1) = p1
            (x2, y2) = p2
            top = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
            bot = math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
            dist = top / bot
            return dist
        
        def colorDist(rgb1, rgb2):
            """ Calculate the distance between two colors (also works for hsv) """
            dr = rgb1[0] - rgb2[0]
            dg = rgb1[1] - rgb2[1]
            db = rgb1[2] - rgb2[2]
        
            dist = math.sqrt(dr * dr + dg * dg + db * db)
            return dist
        
        def pixelDist(p1, p2):
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            d = math.sqrt(dx * dx + dy * dy)
            return d
        
        def poolAndIsolationFilter(gray, initialThresh=0.15, iterations=10):
            filtered = cv.CreateMat(gray.height, gray.width, cv.CV_8UC1)
            cv.Copy(gray, filtered)
            #cv.ShowImage('preErode',filtered)
        
            # Initial Smooth, Erosion, and Threshold
            cv.Smooth(filtered, filtered, cv.CV_GAUSSIAN, 5, -1)
        
            cv.Threshold(filtered, filtered, 255 * initialThresh, 255, cv.CV_THRESH_BINARY)
            
            cv.Erode(filtered, filtered, None, 1);
            #cv.ShowImage('postErode',filtered)
        
            # Repeatedly Dilate and Erode to Merge Nearby Bright Spots
            for i in range(0, iterations):
                cv.Smooth(filtered, filtered, cv.CV_GAUSSIAN, 5, -1)
                cv.Dilate(filtered, filtered, None, 1)
                cv.Erode(filtered, filtered, None, 1)
        
            #cv.ShowImage('postErosionDilation',filtered)
        
            # Smooth, Erode and Threshold a final time
            cv.Smooth(filtered, filtered, cv.CV_GAUSSIAN, 5, -1)
            cv.Erode(filtered, filtered, None, 3)
            
            cv.Threshold(filtered, filtered, 255 * 0.1, 255, cv.CV_THRESH_BINARY)
            
            return filtered
        
        # start code
        grayTestImage = cv.CreateMat(img.height, img.width, cv.CV_8UC1)
        cv.CvtColor(img, grayTestImage, cv.CV_BGR2GRAY)

        testImage = self.vhDeltaProductFilter(grayTestImage)

        results = []
        rh = 1 + testImage.height - self.cube_radius * 2
        rw = 1 + testImage.width - self.cube_radius * 2

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
            cv.Circle(combinedResult, (min_x, min_y), self.cube_radius * 2, maxval, -1)
            (minval, maxval, (min_x, min_y), maxloc) = cv.MinMaxLoc(combinedResult)

        # Find line segments in the original grayscale image
        lines = findLineSegments(grayTestImage)
        ret = []
        for p in results:
            (x, y) = p
            cubeColor = self.avgColor(img, p, self.cube_radius)
            # Match the color to the given template cubes
            minDist = ()
            color = None
            for c in self.templateColors:
                diff = colorDist(cubeColor, self.templateColors[c])
                if diff < minDist:
                    minDist = diff
                    color = c
            # Find the rotation of the cube
            angle = findRotation(lines, p, self.cube_radius * 1.5)
            ret += [(x, y, angle, color)]
        
        return ret

        
    def drawMatch(self, img):
        """ create a match image """
        def getColor(color):
            if color == BLACK:
                return (0, 0, 0)
            elif color == RED:
                return (0, 0, 255)
            elif color == CLEAR:
                return (255, 255, 255)
            elif color == BLUETOOTH:
                return (255, 0, 0)
            
        white = (255, 255, 255)
        dia = 6
        for match in self.matchTemplate(img):
            (x, y, angle, color) = match
            p = (x, y)
            cv.Circle(img, p, dia, getColor(color), -1)
            cv.Circle(img, p, dia, white)
            cv.Circle(img, p, self.cube_radius, white)
            if angle:
                cv.Line(img, p, (int(x + self.cube_radius * numpy.cos(angle)), int(y - self.cube_radius * numpy.sin(angle))), white)

def getImages():
    """ retrieve images into TYPE, FOLDER """
    
    # defs
    class ImageConverter:
        """ provides color and depth images """
        def __init__(self, type=TYPE, folder=FOLDER):
            self.bridge = CvBridge()
            self.image_sub_color = rospy.Subscriber("/camera/rgb/image_color", Image, self.color_callback)
            self.image_sub_depth = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
            self.images_dir = "images/%s/%s" % (type, folder)
            if not os.path.isdir(self.images_dir):
                os.makedirs(self.images_dir)
            self.color_image = None
            self.mono16 = None
            
        def color_callback(self, data):
            try:
                self.color_image = self.bridge.imgmsg_to_cv(data, "bgr8")
            except CvBridgeError, e:
                print e
    
        def depth_callback(self, data):
            try:
                self.mono16 = self.bridge.imgmsg_to_cv(data, "mono16")
            except CvBridgeError, e:
                print e
        
        def save(self, count):
            cv.SaveImage("%s/%s_color.png" % (self.images_dir, count), self.color_image)
            
            my_mono16 = cv.CloneMat(self.mono16)
            depth_image = cv.CreateMat(my_mono16.height, my_mono16.width, cv.CV_8UC3)
            for i in range(my_mono16.height):
                for j in range(my_mono16.width):
                    depth_image[i, j] = (my_mono16[i, j] / (1 << 8), my_mono16[i, j] % (1 << 8), 0)
            cv.SaveImage("%s/%s_depth.png" % (self.images_dir, count), depth_image)
    
        def ready(self):
            return self.color_image and self.mono16
        
    class ImageConverterThread(threading.Thread):
        """ listens to ros messages """
        def __init__(self, threadID, name, counter, ic):
            threading.Thread.__init__(self)
            
            self.threadID = threadID
            self.name = name
            self.counter = counter
            
            self.ic = ic
            rospy.init_node('ImageConverter', anonymous=True)
        
        def run(self):
            rospy.spin()
    
    class ImageSaveThread(threading.Thread):
        """ listens to keyboard input and saves images """
        def __init__(self, threadID, name, counter, ic):
            threading.Thread.__init__(self)
            
            self.threadID = threadID
            self.name = name
            self.counter = counter
    
            self.count = 1
            self.ic = ic
            
        def run(self):
            print "waiting for images"
            while not self.ic.ready():
                print "."
                time.sleep(1)
            
            print "press s followed by <enter> to save"
            
            while 1:
                c = raw_input()
                if c.strip() == "s":
                    self.ic.save(self.count)
                    print "saved %s" % self.count
                    self.count += 1
    
    # code
    ic = ImageConverter()
    
    ic_thread = ImageConverterThread(1, "ic", 1, ic)
    is_thread = ImageSaveThread(2, "is", 2, ic)
    
    ic_thread.start()
    is_thread.start()


def matchImages(test_images=TEST_IMAGES, training_xml=TRAINING_XML,
                 training_image=TRAINING_IMAGE):
    """ match a set of images to a template """
    et = EdgeTemplate(training_xml, training_image)
    for (window_name, color_image, depth_image, result_image, n) in test_images:
        img = cv.LoadImageM(color_image)
        et.drawMatch(img)
        cv.ShowImage(window_name, img)
        # cv.SaveImage(result_image, img)
    
    cv.WaitKey()
    
def matchVideo(training_xml=TRAINING_XML, training_image=TRAINING_IMAGE, camera_index=0):
    et = EdgeTemplate(training_xml, training_image)

    #TODO
