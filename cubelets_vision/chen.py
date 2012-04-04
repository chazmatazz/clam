#!/usr/bin/env python
from CubeletsSURF import *

if __name__ == '__main__':
    try:
        print "Select from the following demos:"
        print "s: Side by Side SURF stable feature video demo"
        print "o: Overlaid SURF stable feature video demo"
        print "m: Stable feature template matching demo"
    
    
        c = raw_input()
    
        if c.strip() == 's':
            surfStableFeatureVideoDemo(0)
        elif c.strip() == 'o':
            surfStableFeatureVideoDemo(1)
        elif c.strip() == 'm':
            surfMatchVideoDemo()
        else:
            "No valid option selected."
        #homographyTest()
        #cameraMatrixTest()
    except rospy.ROSInterruptException: 
        pass
