#!/usr/bin/env python
import CubeletDetection

print "Select from the following demos:"
print "s: Side by Side SURF stable feature video demo"
print "o: Overlaid SURF stable feature video demo"
print "m: Stable feature template matching demo"


c = raw_input()

if c.strip() == 's':
    CubeletDetection.surfStableFeatureVideoDemo(0)
elif c.strip() == 'o':
    CubeletDetection.surfStableFeatureVideoDemo(1)
elif c.strip() == 'm':
    CubeletDetection.surfMatchVideoDemo()
else:
    "No valid option selected."
#homographyTest()
#cameraMatrixTest()