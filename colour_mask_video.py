import cv2
import cv2 as cv
import numpy as np

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

def empty():
    pass

cap = cv.VideoCapture(-1)
width, height = (640,480)
cap.set(3,640)     # width
cap.set(4,480)     # height

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()


    # convert to HSV
    imgHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h_min = 0
    h_max = 170
    s_min = 70
    s_max = 248
    v_min = 131
    v_max = 255

    # Create mask
    lower = np.array([h_min, s_min,v_min])
    upper = np.array([h_max, s_max,v_max])

    # move sliders until orange is achieved
    mask = cv2.inRange(imgHSV,lower, upper)

    imgStack = stackImages(0.6, ([frame, mask]))
    cv2.imshow("stack images", imgStack)

    #Waits for a user input to quit the application
    if cv.waitKey(1) & 0xFF == ord("q"):
        break


# When everything done, release the capture
cap.release()
cv.destroyAllWindows()