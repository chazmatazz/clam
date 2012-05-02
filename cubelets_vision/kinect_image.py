#!/usr/bin/env python
import roslib
roslib.load_manifest('cubelets_vision')
import sys
import rospy
import cv
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import threading
import time
import os
import CubeletDetection

class image_converter:
    """ provides color and depth images """
    def __init__(self, dir):
        self.bridge = CvBridge()
        self.image_sub_color = rospy.Subscriber("/camera/rgb/image_color", Image, self.color_callback)
        self.image_sub_depth = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        self.images_dir = "./images/%s" % dir
        if not os.path.isdir(self.images_dir):
            os.makedirs(self.images_dir)
        self.color_image = None
        self.depth_image = None
        
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
                depth_image[i,j] = (my_mono16[i,j] / (1 << 8), my_mono16[i,j] % (1 << 8), 0)
            
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
        rospy.init_node('image_converter', anonymous=True)
    
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

def main(args):
    print "type directory name followed by <enter>"
    dir = raw_input()
    ic = image_converter(dir.strip())
    ic_thread = ImageConverterThread(1, "ic", 1, ic)
    is_thread = ImageSaveThread(2, "is", 2, ic)
    
    ic_thread.start()
    is_thread.start()
    
    

if __name__ == '__main__':
    main(sys.argv)
    
