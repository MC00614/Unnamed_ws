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
import matplotlib.pyplot as plt
import time

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
    # # Histogram

    def plothistogram(self, image):
        histogram = np.sum(image[image.shape[0]//2:, :], axis=0)
        midpoint = np.int(histogram.shape[0]/2)
        leftbase = np.argmax(histogram[:midpoint])
        rightbase = np.argmax(histogram[midpoint:]) + midpoint
        return leftbase, rightbase

    # left_current 이미지의 왼쪽에 있는 값 중 가장 큰 값을 가진 인덱스
    # good_left window안에 있는 부분만을 저장
    def slide_window_search(self, binary_warped, left_current, right_current):
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        nwindows = 4
        window_height = np.int(binary_warped.shape[0] / nwindows)
        nonzero = binary_warped.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        margin = 100
        minpix = 50
        left_lane = []
        right_lane = []
        color = [0, 255, 0]
        thickness = 2

        for w in range(nwindows):
            win_y_low = binary_warped.shape[0] - (w + 1) * window_height
            win_y_high = binary_warped.shape[0] - w * window_height
            win_xleft_low = left_current - margin     # 왼쪽 window
            win_xleft_high = left_current + margin
            win_xright_low = right_current - margin
            win_xright_high = right_current + margin

            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)
            good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
            good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]

            left_lane.append(good_left)
            right_lane.append(good_right)

            # window의 left_current는 good_left의 길이가 50보다 작으면
            # nonzero_x의 인덱스 good_left의 값을 가지는 인자들의 mean값

            if len(good_left) > minpix:
                left_current = np.int(np.mean(nonzero_x[good_left]))
            if len(good_right) > minpix:
                right_current = np.int(np.mean(nonzero_x[good_right]))
        
        left_lane = np.concatenate(left_lane)     # np.concatenate() --> array를 1차원으로 합침
        right_lane = np.concatenate(right_lane)

        leftx = nonzero_x[left_lane]
        lefty = nonzero_y[left_lane]
        rightx = nonzero_x[right_lane]
        righty = nonzero_y[right_lane]

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        ltx = np.trunc(left_fitx)    # np.trunc()  --> 소수점 부분을 버림
        rtx = np.trunc(right_fitx)

        out_img[nonzero_y[left_lane], nonzero_x[left_lane]] = [255, 0, 0]
        out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = [0, 0, 255]

        ret = {'left_fitx' : ltx, 'right_fitx': rtx, 'ploty': ploty}

        return ret

    # ==================================================================================
    # Draw Line

    def draw_lane_lines(self, original_image, warped_image, Minv, draw_info):
        left_fitx = draw_info['left_fitx']
        right_fitx = draw_info['right_fitx']
        ploty = draw_info['ploty']

        warp_zero = np.zeros_like(warped_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        mean_x = np.mean((left_fitx, right_fitx), axis=0)
        pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

        k = np.int_([pts_mean])

        cv2.fillPoly(color_warp, np.int_([pts]), (255, 255, 255))
        cv2.fillPoly(color_warp, k, (0, 0, 255)
        # 중심의 x,y좌표를 찾아서 원으로 그리기
        cv2.circle(color_warp, (k[0][0][150][0],k[0][0][150][1]), 30, (0,255,0), -1)
        print(k[0][0][150])

        newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
        result = cv2.addWeighted(original_image, 1, newwarp, 0.4, 0)

        return pts_mean, result, k[0][0][150][0]

    def camera_callback(self, _data):
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
        cv_image = self.bridge.compressed_imgmsg_to_cv2(_data)
        
        #cv2.imshow("rect_camera", cv_image)
        #(h, w) = (cv_image.shape[0], cv_image.shape[1])

        # rospy.loginfo("{}".format(h))  cv_image.shape[0]  # 480
        # rospy.loginfo("{}".format(w))  cv_image.shape[1]  # 640

        # ====================================================
        # bird eye view

        source = np.float32([[241, 384], [171, 473], [437, 386], [523, 473]])   # [x,y]
        destination = np.float32([[100, 10], [100, 470], [540, 10], [540, 470]])

        # 변환 행렬을 구하기 위해서 쓰는 함수
        transform_matrix = cv2.getPerspectiveTransform(source, destination)
        # minv값은 마지막에 warpping된 이미지를 복구 역행렬
        minv = cv2.getPerspectiveTransform(destination, source)
        # 변환 행렬값을 적용하여 최종 결과 이미지를 얻을 수 있다.
        _image = cv2.warpPerspective(cv_image, transform_matrix, (w, h))
        #cv2.imshow("warp_image", _image)

        # ===============================================================
        # rgb를 hsv로 변환
        hsv = cv2.cvtColor(_image, cv2.COLOR_BGR2HSV)

        low_H = cv2.getTrackbarPos('low_H', 'roi_Image')
        low_S = cv2.getTrackbarPos('low_S', 'roi_Image')
        low_V = cv2.getTrackbarPos('low_V', 'roi_Image')
        high_H = cv2.getTrackbarPos('high_H', 'roi_Image')
        high_S = cv2.getTrackbarPos('high_S', 'roi_Image')
        high_V = cv2.getTrackbarPos('high_V', 'roi_Image')

        lower = np.array([low_H, low_S, low_V])
        upper = np.array([high_H, high_S, high_V])

        # 어느 정도 threshold값을 확정하면 수치로 고정해서 쓰자
        # lower = np.array([0, 0, 0])
        # upper = np.array([255, 255, 106])

        # masking 적용
        masked = cv2.inRange(hsv, lower, upper)
        # cv2.imshow("lane_image", masked)        
        # cv2.waitKey(1)

        # # ===============================================================

        x = int(masked.shape[1])    # 이미지 가로
        y = int(masked.shape[0])    # 이미지 세로

        # 한붓그리기
        hanboot = np.array(
            [[int(0.1*x), int(y)], [int(0.1*x), int(0.1*y)], [int(0.3*x), int(0.1*y)], [int(0.3*x), int(y)], [int(0.7*x), int(y)], [int(0.7*x), int(0.1*y)], [int(0.9*x), int(0.1*y)], [int(0.9*x), int(y)], [int(0.2*x), int(y)]]
        )

        # 크기같고 0 생성
        hb_mask = np.zeros_like(masked)

        if len(masked.shape) > 2:
            channel_count = masked.shape[2]
            ignore_mask_color = (255, ) * channel_count
        else:
            ignore_mask_color = 255

        # 한붓그리기 영역 생성
        cv2.fillPoly(hb_mask, np.int32([hanboot]), ignore_mask_color)
        # threshold 구분 영역 + 한붓그리기 영역
        masked_image = cv2.bitwise_and(masked, hb_mask)
        #cv2.imshow("lane_image22222", masked_image)        
       
        # _gray = cv2.cvtColor(masked_image, cv2.COLOR_HSV2GRAY)
        # ret, thresh = cv2.threshold(_gray, 160, 255, cv2.THRESH_BINARY)

        ## 선 분포도 조사 histogram
        leftbase, rightbase = self.plothistogram(masked_image)
        # plt.hist(rightbase)
        # plt.show()

        # # ## histogram 기반 window roi 영역
        draw_info = self.slide_window_search(masked_image, leftbase, rightbase)
        # # plt.plot(left_fit)
        # # plt.show()

        # 자 지금부터 욜로 타임~~~~~~~~
        box1 = ImageProcess.ROI(cv_image, scale=1)
        boxes, class_ids, confidences = ImageProcess.YOLO_JUDGE_DT(cv_image, box1)    
    
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(cv_image, label, (x, y + 30), font, 1, color, 1)

        # # ## 원본 이미지에 라인 넣기
        meanPts, result, x_center = self.draw_lane_lines(cv_image, masked_image, minv, draw_info)
        # 중점을 토픽으로 쏴준다.
        self.lane_moment_pub.publish(Int32(x_center))
        cv2.imshow("result", result)

        cv2.waitKey(1)

def nothing():
    pass

# 실제 코드 동작 시, 실행할 코드
def run():
    rospy.init_node("camera_example")     # camera_example이라는 이름으로 노드를 ROS_MASTER에 등록해주는 코드 (이름이 겹치지만 않으면 됨) 
    new_class = CameraReceiver()          # 실제 동작을 담당할 Object
    rospy.spin()                         # ROS Node가 종료되지 않고, Callback 함수를 정상적으로 실행해주기 위한 부분 (코드를 유지하면서, 지속적으로 subscriber를 호출해주는 함수)

if __name__ == '__main__':               # 해당 Python 코드를 직접 실행할 때만 동작하도록 하는 부분
    run()                                # 실제 동작하는 코드


# 점선을 실선으로 인식
# 좌 우 실선의 moment 4 점 구하기
# 그 4점으로 bird eye view

