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

### Local Functions ###


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
current_state = 0
prev_distance = 0
distance = 0
threshold_modifier = 80
distance_thres = 600

### Realsense setup
dc = DepthCamera()

### EasyOCR Setup


### Canny edge setup
x1 = 0

### USB Webcam preprocessing
point = (320,240)

### Main Code
while not rospy.is_shutdown():
    # initialize camerau
    ret, depth_frame, colour_frame = dc.get_frame()
    cv2.circle(colour_frame, point, 3, (0,0,255), 3)

    # current_state = 0, don't move
    # current state = 1, move forward
    # current state = 2, stop again
    # current state = 3, move forward;

    # if significant distance change is detected, ++current state
    prev_distance = distance
    distance = depth_frame[point[1], point[0]]
    difference = float(distance - prev_distance)

    if prev_distance <= distance:
        if difference > 500:
            current_state += 1

        else:
            continue

    # run current state
    if current_state == 0:
        print("state_one")
        # print("state_one")
        cmd_vel.publish(move_cmd_stop)
        r.sleep()

    elif current_state == 1:
        print("state_two")
        # print("state_two")
        cmd_vel.publish(move_cmd_forward)
        r.sleep()

        # if you are close to the door
        if (distance <= distance_thres):
            current_state += 1

    elif current_state == 2:
        print("state_three")
        # print("state_three")
        cmd_vel.publish(move_cmd_stop)
        r.sleep()
        current_state += 1


    elif current_state == 3:
        print("state_four")

        # t0 is the current time
        t0 = rospy.Time.now().secs
        current_angle = 0
        percentage_complete = 0.0
        turn = 5

        while current_angle < turn:
            # Publish the velocity
            print("turning")

            # we need to turn around now
            cmd_vel.publish(move_cmd_right)
            # t1 is the current time
            t1 = rospy.Time.now().secs
            # Calculate current angle
            current_angle = -1 * (move_cmd_right.angular.z) * (t1 - t0)
            # print(current_angle)
            r.sleep()

        # once at correct angle
        cmd_vel.publish(move_cmd_stop)
        r.sleep()
        current_state += 1

    elif current_state == 4:
        print("state_five")
        # stop robot here
        cmd_vel.publish(move_cmd_stop)
        r.sleep()

    elif current_state == 5:
        print("state_six")

        while True:
            ret, depth_frame, colour_frame = dc.get_frame()
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
                    if not_slope < 1 and not_slope > -1:
                        cv.line(colour_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

                    print("pre if")
                    # if the line is too close to the centre, turn away from it
                    if x1 > 320:  # the line is to the right of the centre
                        print("if 1")
                        if (x1 - 320) < 100:
                            cmd_vel.publish(move_cmd_left)
                            r.sleep()

                        else:
                            cmd_vel.publish(move_cmd_forward)
                            r.sleep()

                    else:  # the line is to the left
                        print("if 2")
                        if (320 - x1) < 100:
                            cmd_vel.publish(move_cmd_right)
                            r.sleep()
                        else:
                            cmd_vel.publish(move_cmd_forward)
                            r.sleep()

            else:
                cmd_vel.publish(move_cmd_forward)
                r.sleep()


            # show canny
            cv.imshow("Canny", colour_frame)
            cv2.waitKey(5)

    cv2.imshow("rgb", colour_frame)
    # cv2.imshow("depth", depth_frame)
    cv2.waitKey(5)


