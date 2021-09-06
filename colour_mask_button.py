# colour mask button testing
import cv2
import numpy as np
# Chapter 7, colour detection

import cv2
import numpy as np

print("Package imported")

# stack images function
def stackImages(scale,imgArray):

    rows = len(imgArray)

    cols = len(imgArray[0])

    rowsAvailable = isinstance(imgArray[0], list)

    width = imgArray[0][0].shape[1]

    height = imgArray[0][0].shape[0]

    if rowsAvailable:

        for x in range ( 0, rows):

            for y in range(0, cols):

                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:

                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)

                else:

                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)

                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)

        imageBlank = np.zeros((height, width, 3), np.uint8)

        hor = [imageBlank]*rows

        hor_con = [imageBlank]*rows

        for x in range(0, rows):

            hor[x] = np.hstack(imgArray[x])

        ver = np.vstack(hor)

    else:

        for x in range(0, rows):

            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:

                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)

            else:

                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)

            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)


        hor= np.hstack(imgArray)

        ver = hor

    return ver

path = 'test_images/20210906_105703.jpg'

def empty():
    pass

height = 640
width = 960

# define average position
def getAveragePosition(mask):
    xAverage = 0
    yAverage = 0
    count = 0
    resolution = 25
    for y in range(0, height, resolution):
        for x in range(0, width, resolution):
            if mask[y][x] == 255:
                xAverage += x
                yAverage += y
                count += 1

    if count > 0:
        xAverage = xAverage / count
        yAverage = yAverage / count
    return(xAverage, yAverage)

# Create trackbars to find minimum and maximum numbers for orange
cv2.namedWindow("TrackBars")

# name has to be the same
cv2.resizeWindow("TrackBars", 640, 300)
cv2.createTrackbar("Hue Min", "TrackBars", 147, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars",179, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 82, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 236, 255,empty)
cv2.createTrackbar("Val Min", "TrackBars", 127, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255,empty)

# Make TrackBars do something
while True:
    og = cv2.imread(path)
    img = cv2.resize(og, (960, 640))

    # convert to HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # get hue min from trackbars
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min, h_max, s_min, s_max, v_min, v_max)

    # Create mask
    lower = np.array([h_min, s_min,v_min])
    upper = np.array([h_max, s_max,v_max])

    # move sliders until orange is achieved
    mask = cv2.inRange(imgHSV,lower, upper)
    mask_copy = mask.copy()

    # test average white pixel (source: Miekal reves)
    xAverage, yAverage = getAveragePosition(mask_copy)
    cv2.circle(mask_copy, (round(xAverage), round(yAverage)), 20, (255, 0, 0), -1)

    # create new image with mask
    imgResult = cv2.bitwise_and(img, img, mask= mask)

    # cv2.imshow("Original", img)
    # cv2.imshow("HSV", imgHSV)
    # cv2.imshow("mask", mask)
    # cv2.imshow("result", imgResult)

    imgStack = stackImages(0.6, ([img, imgHSV], [mask_copy, imgResult]))
    cv2.imshow("images", imgStack)
    cv2.waitKey(1)