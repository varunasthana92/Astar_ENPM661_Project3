#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
# from cv_bridge import CvBridge
# import cv2
# import os
import numpy as np
from astar import astar

class Node(object):
    def __init__(self):
        # Params
        self.data = Twist()
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(0.5)

        # Publishers
        self.pub = rospy.Publisher("/cmd_vel_mux/input/navi", Twist, queue_size=10000)
        self.loop_rate.sleep()

    def pubData(self, lin, rot):

        self.data.angular.z= rot
        self.data.linear.x = lin
        #rospy.spin()
        
            # rospy.loginfo('publishing cmd data')
        # for i in range(10):
        self.pub.publish(self.data)
        self.loop_rate.sleep()

if __name__ == '__main__':
    
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--explore', default=0, help='Set to 1 if want to see all explored nodes animation (default: 0)')
    # Parser.add_argument('--inp', default=0, help='Set to 1 to provide user input data (default: 0)')

    # Args = Parser.parse_args()
    rospy.init_node("botmove", anonymous=False)
    init_x = rospy.get_param('~init_x')
    init_y = rospy.get_param('~init_y')
    init_theta = rospy.get_param('~init_t')
    clrn = [float(rospy.get_param('~clr'))]
    my_node = Node()
    initCord = [init_x, init_y, init_theta]
    wheel_rad = 0.038 # 0.076/2
    wheel_dis = 0.354
    actionSet = astar(initCord, clrn)
    if(len(actionSet) != 0):
        for move in actionSet:
            ul = move[0]
            ur = move[1]
            x = (wheel_rad/2)*(ul+ur)
            v_diff = (wheel_rad/2)*(ur-ul)
            z = v_diff*2/wheel_dis
            my_node.pubData(0.812*x,0.965*z)
            my_node.pubData(0.812*x,0.965*z)
