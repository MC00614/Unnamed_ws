#!/usr/bin/env python
#-*- coding: utf-8 -*-

import ImageProcess

import rospy
import cv2
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge                  # ROS Image를 OpenCV의 Image로 변경하기 위해 필요한 코드
import numpy as np
from std_msgs.msg import Int32
from std_msgs.msg import Float64
from std_msgs.msg import Int8MultiArray

frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
classes = []
with open("YOLODATA/obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

class CameraReceiver():
    def __init__(self):
        # rospy.loginfo("Camera Receiver object is created")
        # 토픽 수신
        rospy.Subscriber("usb_cam/image_rect_color/compressed", CompressedImage, self.camera_callback)
        self.bridge = CvBridge()
        self.initialized = False
        # 토픽 발신
        self.lane_moment_pub = rospy.Publisher("center_lane_moment_x", Int32, queue_size=3)

    # # ==================================================================================
    def camera_callback(self, data):
        if self.initialized == False:       # 아래의 코드는 처음 실행했을 때, 한번만 실행되어야하므로 self.initialized를 사용해서 처음에만 실행되도록 짜준 코드
            cv2.namedWindow("roi_Image", cv2.WINDOW_NORMAL)       # Simulator_Image를 원하는 크기로 조정가능하도록 만드는 코드
            # 흰색 차선을 검출할 때, 이미지를 보면서 트랙바를 움직이면서 흰색선이 검출되는 정도를 파악하고 코드안에 그 수치를 넣어준다.
            cv2.createTrackbar('low_H', 'roi_Image', 0, 255, nothing)    # Trackbar 만들기
            cv2.createTrackbar('low_S', 'roi_Image', 0, 255, nothing)
            cv2.createTrackbar('low_V', 'roi_Image', 155, 255, nothing)
            cv2.createTrackbar('high_H', 'roi_Image', 255, 255, nothing)    
            cv2.createTrackbar('high_S', 'roi_Image', 255, 255, nothing)
            cv2.createTrackbar('high_V', 'roi_Image', 255, 255, nothing)
            self.initialized = True  # 두 번 다시 여기 들어오지 않도록 처리

        #============== 카메라 imgmsg --> cv2 =========================
        cv_image = self.bridge.compressed_imgmsg_to_cv2(data)

        source = np.float32([[241, 384], [171, 473], [437, 386], [523, 473]])   # [x,y]
        destination = np.float32([[100, 10], [100, 470], [540, 10], [540, 470]])


        ################################################################################
        # 이미지 처리(그림은 나중에 한번에 그림)
        box_candis = ImageProcess.TS_Candidates(cv_image)
        
        boxes, class_ids, confidences = ImageProcess.YOLO_JUDGE_DT(cv_image, box_candis)
        
        X1, Y1, X2, Y2 = ImageProcess.StopLine(cv_image)
        
        thresh, minverse, draw_info = ImageProcess.LANEBYHLS(cv_image, source, destination)
        ################################################################################


        ################################################################################
        # 원본 위에 그림(상자)그리기(단계별로 yolo->sl->lane_hls)
        img_yolo = ImageProcess.YOLO_Labeling(cv_image, boxes, class_ids, confidences)

        img_yolo_sl = ImageProcess.DrawStopLine(img_yolo, X1, Y1, X2, Y2)

        meanPts, result  = ImageProcess.drawLANEBYHLS(img_yolo_sl, thresh, minverse, draw_info)
        ################################################################################

        # 중점을 토픽으로 쏴준다.
        self.lane_moment_pub.publish(Int32(meanPts))
        cv2.imshow("result", result)

        cv2.waitKey(1)

def nothing():
    pass

# 실제 코드 동작 시, 실행할 코드
def run():
    rospy.init_node("Detection")     # camera_example이라는 이름으로 노드를 ROS_MASTER에 등록해주는 코드 (이름이 겹치지만 않으면 됨) 
    CameraReceiver()          # 실제 동작을 담당할 Object
    rospy.spin()                         # ROS Node가 종료되지 않고, Callback 함수를 정상적으로 실행해주기 위한 부분 (코드를 유지하면서, 지속적으로 subscriber를 호출해주는 함수)

if __name__ == '__main__':
    run()
