#!/usr/bin/env python
import CubeletsSURF

if __name__ == '__main__':
    try:
        CubeletsSURF.matchTemplateImage()
    except rospy.ROSInterruptException: 
        pass
