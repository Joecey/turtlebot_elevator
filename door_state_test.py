# code to test door state
import cv2
from numpy import *
print("packages imported")

test_imgs = ['test_images/open_door_night.jpg', 'test_images/closed_door_night.jpg',
             'test_images/open_door_day.jpg', 'test_images/closed_door_day.jpg']
#
for imgFile in test_imgs:
    img = cv2.imread(imgFile)
    height, width, channels = img.shape
    mask = zeros((height+2, width+2), uint8)

    #the starting pixel for the floodFill
    start_pixel = (510,110)


    #maximum distance to start pixel:
    diff = (2,2,2)

    # detect similar pixel colours and fill
    retval = cv2.floodFill(img, mask, start_pixel, (0,255,0), diff, diff)

    # print original circle
    cv2.circle(img, start_pixel, 3, (0, 0, 255), thickness=5)


    print(retval)
    cv2.imshow("result", img)
    cv2.waitKey(1000)

    #check the size of the floodfilled area, if its large the door is closed:
    # if retval > 10000:
    #     print(imgFile) + ": garage door closed"
    # else:
    #     print(imgFile)+ ": garage door open"

    # cv2.imwrite(imgFile.replace(".jpg", "") + "_result.jpg", img)