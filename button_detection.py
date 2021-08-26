# algorithm for button detection
# import packages
import cv2 as cv
import numpy as np
import pytesseract
import tesserocr
from PIL import Image
import matplotlib.pyplot as plt
import imutils
from scipy import ndimage

example_img = 'sample_text_images/close_up_2.jpg'

test_imgs = ['elevator_button_images/1.jpeg', 'elevator_button_images/2.jpg',
             'elevator_button_images/3.jpg', 'elevator_button_images/4.jpg']

# Functions
# stack images function
# You can copy this function to stack multiple image sources in the one window - Joe
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

                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)

                else:

                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)

                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)

        imageBlank = np.zeros((height, width, 3), np.uint8)

        hor = [imageBlank]*rows

        hor_con = [imageBlank]*rows

        for x in range(0, rows):

            hor[x] = np.hstack(imgArray[x])

        ver = np.vstack(hor)

    else:

        for x in range(0, rows):

            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:

                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)

            else:

                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)

            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)


        hor= np.hstack(imgArray)

        ver = hor

    return ver #if I want two videos in the one window


# Testing
img = cv.imread(example_img)

scale_percent = 20 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
copy = img.copy()

# create black mask for bianry mask
# Prepare a black canvas:
mask = np.zeros((height, width), dtype=np.uint8)


#
# alpha = 1 # Contrast control (1.0-3.0)
# beta = 20 # Brightness control (0-100)

# copy = cv.convertScaleAbs(copy, alpha=alpha, beta=beta)

# # Apply hough transform
# # detect circles in the image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1.2, 50)

# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    # loop over the (x, y) coordinates and radius of the circles
for (x, y, r) in circles:

    # add white circle in binary mask
    cv.circle(mask, (x,y), r, (255,255,255), thickness=-1)

    # draw the circle in the output image, then draw a rectangle
    # corresponding to the center of the circle
    # cv.circle(copy, (x, y), r, (0, 255, 0), 4)
    # cv.rectangle(copy, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)


# show the output image
# apply binary mask onto image

res = cv.bitwise_or(copy, copy, mask = mask)

# threshold the warped image, then apply a series of morphological
# operations to cleanup the thresholded image
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 2))
res = cv.morphologyEx(res, cv.MORPH_OPEN, kernel)
gray_res = cv.cvtColor(res, cv.COLOR_BGR2GRAY)

thresh1 = cv.threshold(gray_res, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

# use adaptive thresholding
thresh2 = cv.adaptiveThreshold(gray_res,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY_INV,11,2)
thresh3 = cv.adaptiveThreshold(gray_res,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY_INV,11,2)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 3))
thresh2 = cv.morphologyEx(thresh2 , cv.MORPH_OPEN, kernel)
thresh3 = cv.morphologyEx(thresh3 , cv.MORPH_OPEN, kernel)


# Adding custom options
custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
print(pytesseract.image_to_string(thresh2, config=custom_config))

# img_stack = stackImages(0.6, ([gray_res, thresh1], [thresh2, thresh3]))

cv.imshow("thresh3",thresh3)
cv.imshow("thresh2", thresh2)
# cv.imshow("binary", mask)
cv.waitKey(0)
cv.destroyAllWindows()


# Create image stack
# img_stack = stackImages(0.9, ([img, output]))


