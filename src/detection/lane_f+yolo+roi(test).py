#!/usr/bin/env python
#-*-coding: utf-8 -*-

import ImageProcess
#import rospy
import cv2
import youtube_dl
import pafy
import numpy as np
import matplotlib.pyplot as plt
import time

# ==================================================================================
# youtube 영상 불러오기
# https://velog.io/@bangsy/Python-OpenCV-4      설명
# https://www.youtube.com/watch?v=YsPdvvixYfo&t=0s
# https://www.youtube.com/watch?v=ipyzW38sHg0

url = 'https://www.youtube.com/watch?v=ipyzW38sHg0'
video = pafy.new(url)
best = video.getbest(preftype = 'mp4')

# cap = cv2.VideoCapture(best.url)

cap = cv2.VideoCapture(0)

# cap = cv2.VideoCapture('project_video.mp4')

frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
classes = []
with open("YOLODATA/obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def dt(img):
    (height, width) = (img.shape[0], img.shape[1])
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 1, color, 1)

    

def wrapping(image):
    (h, w) = (image.shape[0], image.shape[1])

    source = np.float32([[w // 2 - 30, h * 0.53], [w // 2 +60, h * 0.53], [w * 0.3, h], [w, h]])
    destination = np.float32([[0, 0], [w-350, 0], [400, h], [w-150, h]])

    # source = np.float32([[560, 450], [275, 665], [720, 450], [1150, 665]])
    # destination = np.float32([[200, 10], [200, 720], [1060, 10], [1060, 720]])

    # 변환 행렬을 구하기 위해서 쓰는 함수
    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    # minv값은 마지막에 warpping된 이미지를 다시 원근감을 
    minv = cv2.getPerspectiveTransform(destination, source)
    # 변환 행렬값을 적용하여 최종 결과 이미지를 얻을 수 있다.
    _image = cv2.warpPerspective(image, transform_matrix, (w, h))

    return _image, minv

# ==================================================================================
# Color Filter(HLS사용)

def color_filter(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    lower = np.array([20, 150, 20])
    upper = np.array([255, 255, 255])

    yellow_lower = np.array([0, 85, 81])
    yellow_upper = np.array([190, 255, 255])

    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hls, lower, upper)
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)

    return masked

# ==================================================================================
# ROI

def roi(image):
    x = int(image.shape[1])
    y = int(image.shape[0])

    _shape = np.array(
        [[int(0.1*x), int(y)], [int(0.1*x), int(0.1*y)], [int(0.4*x), int(0.1*y)], [int(0.4*x), int(y)], [int(0.7*x), int(y)], [int(0.7*x), int(0.1*y)], [int(0.9*x), int(0.1*y)], [int(0.9*x), int(y)], [int(0.2*x), int(y)]]
    )

    mask = np.zeros_like(image)

    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255, ) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, np.int32([_shape]), ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

# ==================================================================================
# Histogram

def plothistogram(image):
    histogram = np.sum(image[image.shape[0]//2:, :], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftbase = np.argmax(histogram[:midpoint])
    rightbase = np.argmax(histogram[midpoint:]) + midpoint

    return leftbase, rightbase


# left_current 이미지의 왼쪽에 있는 값 중 가장 큰 값을 가진 인덱스
# good_left window안에 있는 부분만을 저장
def slide_window_search(binary_warped, left_current, right_current):
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

    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color = 'yellow')
    # plt.plot(right_fitx, ploty, color = 'yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()

    ret = {'left_fitx' : ltx, 'right_fitx': rtx, 'ploty': ploty}

    return ret

# ==================================================================================
# Draw Line

def draw_lane_lines(original_image, warped_image, Minv, draw_info):
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

    cv2.fillPoly(color_warp, np.int_([pts]), (255, 255, 255))
    cv2.fillPoly(color_warp, np.int_([pts_mean]), (0, 0, 255))

    newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
    result = cv2.addWeighted(original_image, 1, newwarp, 0.4, 0)

    return pts_mean, result

while True:
    retval, img = cap.read()
    if not retval:
        break

    #cv2.imshow('1', img)

    ## 조감도 wrapped img
    wrapped_img, minverse = wrapping(img)
    #cv2.imshow('wrapped', wrapped_img)
    
    ## 조감도 필터링
    w_f_img = color_filter(wrapped_img)
    #cv2.imshow('w_f_img', w_f_img)

    ## 조감도 필터링 자르기
    w_f_r_img = roi(w_f_img)
    #cv2.imshow('w_f_r_img', w_f_r_img)

    ## 조감도 선따기 wrapped img threshold
    _gray = cv2.cvtColor(w_f_r_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(_gray, 160, 255, cv2.THRESH_BINARY)    
    #cv2.imshow('threshold', thresh)

    ## 선 분포도 조사 histogram
    leftbase, rightbase = plothistogram(thresh)
    # plt.hist(rightbase)
    # plt.show()
    
    # ## histogram 기반 window roi 영역
    draw_info = slide_window_search(thresh, leftbase, rightbase)
    # plt.plot(left_fit)
    # plt.show()

    box1 = ImageProcess.ROI(img, scale=1)
    boxes, class_ids, confidences = ImageProcess.YOLO_JUDGE_DT(img, box1)    
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 1, color, 1)

    # ## 원본 이미지에 라인 넣기
    meanPts, result = draw_lane_lines(img, thresh, minverse, draw_info)
    #meanPts, result = draw_lane_lines(img, thresh, minverse, draw_info)
    cv2.imshow("result", result)

    # ## 동영상 녹화
    # out1.write(result)

    key = cv2.waitKey(25)
    if key == 27:
        break

if cap.isOpened():
    cap.release()

cv2.destroyAllWindows()






