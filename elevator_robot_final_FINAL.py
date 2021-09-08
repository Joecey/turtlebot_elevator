#!/usr/bin/env python3


# code for all computer vision tasks and turtlebot movement
# import packages
import rospy
from geometry_msgs.msg import Twist
import cv2
import cv2 as cv
import pyttsx3
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2
from realsense_depth import *
print("packages imported")

height = 480
width = 640

### Local Functions ###
# Get average point

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

#### Intialization for rospy ####
rospy.init_node('rotate', anonymous=False)
print("node made")
# tell user how to stop TurtleBot
rospy.loginfo("To stop TurtleBot CTRL + C")


# Create a publisher which can "talk" to TurtleBot and tell it to move
# Tip: You may need to change cmd_vel_mux/input/navi to /cmd_vel if you're not using TurtleBot2
cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)

# TurtleBot will stop if we don't keep telling it to move.  How often should we tell it to move? 10 HZ
# 0.1 second = 10 hz
r = rospy.Rate(10);

# Twist is a datatype for velocity
move_cmd_right = Twist()
move_cmd_right.linear.x = 0.0
move_cmd_right.angular.z = -0.3

# turn left
move_cmd_left = Twist()
move_cmd_left.linear.x = 0.0
move_cmd_left.angular.z = 0.3
#
# move stop
move_cmd_stop = Twist()
move_cmd_stop.linear.x = 0
move_cmd_stop.angular.z = 0

# forward
move_cmd_forward = Twist()
move_cmd_forward.linear.x = 0.2
move_cmd_forward.angular.z = 0

# Global command for robot
string_command = ""

### FSM variable setup
current_state = 1
prev_distance = 0
distance = 0
threshold_modifier = 80
distance_thres = 700

### Realsense setup
dc = DepthCamera()

### colour mask

h_min = 147
h_max = 179
s_min = 82
s_max = 236
v_min = 127
v_max = 255


### Canny edge setup
x1 = 0

### USB Webcam preprocessing
point = (300,300)


### Main Code
while not rospy.is_shutdown():
    # initialize camerau
    ret, depth_frame, colour_frame = dc.get_frame()
    cv2.circle(colour_frame, point, 3, (0,0,255), 3)

    # # Apply hough transform
    # # detect circles in the image
    gray = cv.cvtColor(colour_frame, cv.COLOR_BGR2GRAY)
    imgHSV = cv2.cvtColor(colour_frame, cv2.COLOR_BGR2HSV)

    # current_state = 0, don't move
    # current state = 1, move forward
    # current state = 2, stop again
    # current state = 3, turn to buttons;
    # current state = 4, wait for button to be pressed;
    # current state = 5, turn to door;
    # current state = 6, exit lift;

    # if significant distance change is detected, ++current state
    prev_distance = distance
    distance = depth_frame[point[1], point[0]]
    # print(distance)
    difference = float(distance - prev_distance)

    if prev_distance <= distance:
        if difference > 500:
            current_state += 1

        else:
            continue

    # run current state
    if current_state == 1:
        print("state_one")
        # print("state_one")
        cmd_vel.publish(move_cmd_stop)
        rospy.Rate(10).sleep()

    elif current_state == 2:
        print("state_two")
        # print("state_two")
        cmd_vel.publish(move_cmd_forward)
        rospy.Rate(10).sleep()



        # if you are close to the door
        if (distance <= distance_thres):
            current_state += 1

    elif current_state == 3:
        print("state_three")
        cmd_vel.publish(move_cmd_stop)
        r.sleep()
        current_state += 1


    elif current_state == 4:
        print("state_four")    #turning to find buttons PUT JOE THINGS HERE?
        # t0 is the current time
        t0 = rospy.Time.now().secs
        current_angle = 0
        percentage_complete = 0.0
        turn = 4.1  # this should be a quarter turn to face the doors

        while current_angle < turn:
            # Publish the velocity

            # we need to turn around now
            cmd_vel.publish(move_cmd_left)
            # t1 is the current time
            t1 = rospy.Time.now().secs
            # Calculate current angle
            current_angle = (move_cmd_left.angular.z) * (t1 - t0)
            print(current_angle)
            # print(current_angle)
            r.sleep()

        # once at correct angle
        cmd_vel.publish(move_cmd_stop)
        r.sleep()
        current_state += 1

    elif current_state == 5:
        print("state_five")   #looking at buttons time (JOE'S STUFF HERE)
        # stop robot here
        cmd_vel.publish(move_cmd_stop)


        # check if button is pressed
        # Apply Hough Transform
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1.2, 80)

        # Create mask
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        # apply mask
        press_mask = cv2.inRange(imgHSV, lower, upper)

        # get average point of the red ring caused by pressing button
        xAverage, yAverage = getAveragePosition(press_mask)

        if circles is not None:
            shortest_distance = 1000
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

            for (x, y, r) in circles:
                # add white circle in binary mask
                # cv.circle(mask, (x,y), r, (255,255,255), thickness=-1)

                #### ADD TEXT DETECTION HERE ####

                # test if button is pressed before checking number
                # test distance between average point and each centre
                temp_1 = (round(xAverage) - x) ^ 2
                temp_2 = (round(yAverage) - y) ^ 2

                if temp_1 >= 0 and temp_2 >=0:
                    distance = temp_1 + temp_2
                    # if next circle distance from blue dot is less than previous
                    if distance < shortest_distance:
                        shortest_distance = distance

                    # print(shortest_distance)

                    # if the distance between the blue circle and the closest circle is below
                    # threshold, this indicates that the button is pressed
                    if shortest_distance < 100:
                        print("button is pressed")
                        current_state += 1


    elif current_state == 6:
        print("state_six")  # turning to face doors


        # t0 is the current time
        t0 = rospy.Time.now().secs
        current_angle = 0
        percentage_complete = 0.0
        turn = 3.14              #this should be a quarter turn to face the doors

        while current_angle < turn:
            # Publish the velocity


            # we need to turn around now
            cmd_vel.publish(move_cmd_left)
            # t1 is the current time
            t1 = rospy.Time.now().secs
            # Calculate current angle
            current_angle =  (move_cmd_left.angular.z) * (t1 - t0)
            print(current_angle)
            # print(current_angle)
            # r.sleep()

        # once at correct angle
        cmd_vel.publish(move_cmd_stop)
        # r.sleep()
        print("stop")
        current_state += 1

    elif current_state == 7:
        print("state_seven")
        # print("state_one")
        cmd_vel.publish(move_cmd_stop)
        rospy.Rate(10).sleep()
        # current_state+=1

    elif current_state == 8:
        print("state_eight")
        # cv2.destroyAllWindows()

        while True:
            ret, depth_frame, colour_frame = dc.get_frame()
            distance = depth_frame[point[1], point[0]]
            grey = cv.cvtColor(colour_frame, cv.COLOR_BGR2GRAY)
            edges = cv.Canny(grey, 100, 200)
            lines = cv.HoughLines(edges, 1, (np.pi / 180), 200)

            if lines is not None:
                for line in lines:
                    rho, theta = line[0]
                    # changing polar coordinates to cartesian
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho

                    # not sure of logic behind next bit - found in this video https://www.youtube.com/watch?v=gbL3XKOiBvw
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * a)
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * a)

                    # figuring out slope so just finding vertical lines
                    not_slope = (x2 - x1) / (y2 - y1)
                    if not_slope < 0.5 and not_slope > -0.5:
                        cv.line(colour_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

                    # if the line is too close to the centre, turn away from it
                    if x1 > 320:  # the line is to the right of the centre
                        print("if 1")
                        if (x1 - 320) < 100:
                            cmd_vel.publish(move_cmd_left)
                            rospy.Rate(10).sleep()

                        else:
                            if distance > 300:
                                cmd_vel.publish(move_cmd_forward)
                                rospy.Rate(10).sleep()

                    else:  # the line is to the left
                        print("if 2")
                        if (320 - x1) < 100:
                            cmd_vel.publish(move_cmd_right)
                            rospy.Rate(10).sleep()
                        else:
                            if distance > 300:
                                cmd_vel.publish(move_cmd_forward)
                                rospy.Rate(10).sleep()

            else:
                if distance > 300:
                    cmd_vel.publish(move_cmd_forward)
                    rospy.Rate(10).sleep()


            # show canny
            cv.imshow("Canny", colour_frame)
            cv2.waitKey(5)

    cv2.imshow("rgb", colour_frame)
    # cv2.imshow("depth", depth_frame)
    cv2.waitKey(5)