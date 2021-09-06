#!/usr/bin/env python3

# cool button tracker thing
# webcam for button detection
# uses pytorch with CUDA 10.0 i.e GPU
# import packages
import time
import cv2 as cv
import cv2
import numpy as np

# OCR stuff
from easyocr import Reader

# realsense
import pyrealsense2
from realsense_depth import *

numbers_possible = ["1", "2", "3", "4",
                    "5", "6", "7", "8", "9"]

# clean up non-ascii text
def cleanup_text(text):
    # strip out non-ASCII text so we can draw the text on the image
    # using OpenCV
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

# setup webcam (we will probably be using a different camera module in the future)
# set to 1/2 to get external usb recent camera
cap = cv.VideoCapture(0)
width, height = (640,480)
cap.set(3,640)     # width
cap.set(4,480)     # height
cap.set(10, 100)     # brightness
cap.set(11, 20)     # contrast

# other webcam parameters


# setup easyocr reader
# OCR cropped image, langauage set to english, default gpu=True (CUDA 10.0)
# default threshold = 0.7
reader = Reader(['en'])

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

# setup realsense
# dc = DepthCamera()

while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()

    # capture frame by frame (realsense)
    # ret, depth_frame, frame = dc.get_frame()

    # record frame rate (avg. 20 to 30 fps)
    # time when we finish processing for this frame
    new_frame_time = time.time()

    # Calculating the fps
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = str(round(fps))
    cv.putText(frame, fps + "FPS", (7, 70), cv.FONT_HERSHEY_PLAIN, 3, (100, 255, 0), 3, cv.LINE_AA)

    # create binary mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # # Apply hough transform
    # # detect circles in the image
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1.2, 50)

    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            # add white circle in binary mask
            # cv.circle(mask, (x,y), r, (255,255,255), thickness=-1)

            top_left = (x - round(0.4 * r), y - round(0.9 * r))
            bottom_right = (x + round(0.4 * r), y + round(0.25 * r))

            # isolate numbers?
            cv.rectangle(mask, top_left, bottom_right, (255, 255, 255), thickness=-1)

            # apply image crop
            cropped = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            # read from the cropped text
            if (cropped is not None):
                results = reader.readtext(cropped, allowlist='0123456789')

            if results is not None:
                # loop over the results
                for (bbox, text, prob) in results:
                    # display the OCR'd text and associated probability
                    # print("[INFO] {:.4f}: {}".format(prob, text))
                    # unpack the bounding box
                    (tl, tr, br, bl) = bbox
                    tl = (int(tl[0]), int(tl[1]))
                    tr = (int(tr[0]), int(tr[1]))
                    br = (int(br[0]), int(br[1]))
                    bl = (int(bl[0]), int(bl[1]))
                    # cleanup the text and draw the box surrounding the text along
                    # with the OCR'd text itself
                    text = cleanup_text(text)

                    # for each text, write it on copy image
                    cv.putText(frame, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=2)

    # result of crop
    res = cv.bitwise_or(frame, frame, mask=mask)

    # Display the resulting frame
    cv.imshow("no mask", frame)
    cv.imshow("preview",res)
    cv.waitKey(5)

    #Waits for a user input to quit the application
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()