import cv2 as cv
import numpy as np

#img = cv.imread('test_images/lift-doors.jpg', cv.IMREAD_GRAYSCALE)

#webcam
cam = cv.VideoCapture(0)
cam.set(11, 0)

ret, img = cam.read()

# Draw centre circle on frame
cv.circle(img, (320, 240), 3, (0, 0, 255), 3)

while True:
    ret, img = cam.read()
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(grey, 100, 200)
    lines = cv.HoughLines(edges, 1, (np.pi / 180), 200)
    print(str(type(lines)))

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            #changing polar coordinates to cartesian
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            #not sure of logic behind next bit - found in this video https://www.youtube.com/watch?v=gbL3XKOiBvw
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)

            #figuring out slope so just finding vertical lines
            not_slope = (x2 - x1)/(y2 - y1)
            if not_slope < 1 and not_slope > -1:
                cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

    cv.imshow("Original", img)
    cv.imshow("grey", grey)
    cv.imshow("Canny", edges)
    cv.waitKey(5)

    #if the line is too close to the centre, turn away from it
    if x1 > 320: #the line is to the right of the centre
        if (x1 - 320) < 50:
            print("turn left")
    else: #the line is to the left
        if (320 - x1) < 50:
            print("turn right")



    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
cv.VideoCapture(0).release()