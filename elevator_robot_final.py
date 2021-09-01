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
move_cmd_forward.linear.x = 0.1
move_cmd_forward.angular.z = 0

# Global command for robot
string_command = ""

### FSM variable setup
current_state = 0

### Realsense setup


### EasyOCR Setup


### USB Webcam preprocessing


### Main Code
while not rospy.is_shutdown():
    # current_state = 0, don't move
    # current state = 1, move forward
    # current state = 2, stop again
    # current state = 3, move forward;

    # if significant distance change is detected, ++current state

    # run current state
    if current_state == 0:
        print("state_one")
    elif current_state == 1:
        print("state_two")
    elif current_state == 2:
        print("state_three")
    elif current_state == 3:
        print("state_four")

    string_command = "forward"
    cmd_vel.publish(move_cmd_forward)
    r.sleep()
