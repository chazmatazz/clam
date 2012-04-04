#!/usr/bin/env python
from CubeletsSURF import *

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