# code for all computer vision tasks and turtlebot movement
# import packages
import rospy
from geometry_msgs.msg import Twist
import cv2
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
r = rospy.Rate(5);

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
            current_state = current_state + 1

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
            current_state = current_state + 1

    elif current_state == 2:
        print("state_three")
        # print("state_three")
        cmd_vel.publish(move_cmd_stop)
        r.sleep()

    elif current_state == 3:
        print("state_four")
        cmd_vel.publish(move_cmd_forward)
        r.sleep()

    cv2.imshow("rgb", colour_frame)
    # cv2.imshow("depth", depth_frame)
    cv2.waitKey(5)


