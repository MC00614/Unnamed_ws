#!/usr/bin/python3
import rospy
import numpy as np
import json
from std_msgs.msg import String, Float32
from std_msgs.msg import Int32, Int32MultiArray


class Controller():
    def __init__(self):
        rate = rospy.Rate(10) # 10hz
        self.control = rospy.Publisher('/automobile/command', String, queue_size=1)

        rospy.Subscriber("/automobile/Detection/SIZE", Int32MultiArray, self.SIZE_callback)
        rospy.Subscriber("/automobile/Detection/LANE", Int32MultiArray, self.LANE_callback)
        rospy.Subscriber("/automobile/Detection/OBJECT", Int32MultiArray, self.OBJECT_callback)
        rospy.Subscriber("/automobile/Detection/SL", Int32, self.SL_callback)
        
        angle = self.controller(self.lane, self.object, self.sl)
        self.control.publish(angle)
        rate.sleep()
        speed = drive_speed(0.5)
        self.control.publish(speed)
        rate.sleep()


    def SIZE_callback(self,data):
        self.size = data

    def LANE_callback(self, data):
        self.lane = data

    def OBJECT_callback(self, data):
        self.object = data

    def SL_callback(self, data):
        self.sl = data

    def controller(size, lane, object, sl):
        if lane[0]!=False and lane[1]!=False:
            middle = np.average(lane) - size[1]//2
            angle = drive_angle(middle*(np.pi/180)*0.1)
            return angle

def drive_speed(speed=0.0):
    data = {}
    data['action']        =  '1'
    data['speed']         =  float(speed)
    reference = json.dumps(data)
    return reference

def drive_angle(angle=0.0):
    data = {}
    data['action']        =  '2'
    data['steerAngle']    =  float(angle)
    reference = json.dumps(data)
    return reference


def run():
    rospy.init_node("Controller")
    Controller()
    rospy.spin()

if __name__ == '__main__':
    run()