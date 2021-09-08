import cv2
from realsense_depth import *

dc = DepthCamera()

while(True):
    ret, depth_frame, frame = dc.get_frame()
    cv2.circle(frame, (120, 400), 15, (255,0,0), -1)
    cv2.imshow("bitch", frame)
    cv2.waitKey(1)