import os
# 경로문제 안생기게 바꿈
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import ImageProcess
import numpy as np
import cv2

def nothing(a=1):
    return

cap = cv2.VideoCapture('./TESTDATA/asd.mp4')
initialized = False

while True:
    retval, img = cap.read()
    if not retval:
        break
    
    
    
    if initialized == False:       # 아래의 코드는 처음 실행했을 때, 한번만 실행되어야하므로 self.initialized를 사용해서 처음에만 실행되도록 짜준 코드
            cv2.namedWindow("roi_Image", cv2.WINDOW_NORMAL)       # Simulator_Image를 원하는 크기로 조정가능하도록 만드는 코드
            # 흰색 차선을 검출할 때, 이미지를 보면서 트랙바를 움직이면서 흰색선이 검출되는 정도를 파악하고 코드안에 그 수치를 넣어준다.
            cv2.createTrackbar('low_H', 'roi_Image', 0, 255, nothing)    # Trackbar 만들기
            cv2.createTrackbar('low_S', 'roi_Image', 0, 255, nothing)
            cv2.createTrackbar('low_V', 'roi_Image', 155, 255, nothing)
            cv2.createTrackbar('high_H', 'roi_Image', 255, 255, nothing)    
            cv2.createTrackbar('high_S', 'roi_Image', 255, 255, nothing)
            cv2.createTrackbar('high_V', 'roi_Image', 255, 255, nothing)
            initialized = True  # 두 번 다시 여기 들어오지 않도록 처리

    (h, w) = (img.shape[0], img.shape[1])
    # source = np.float32([[241, 384], [171, 473], [437, 386], [523, 473]])   # [x,y]
    # destination = np.float32([[100, 10], [100, 470], [540, 10], [540, 470]])
    
    source = np.float32([[w // 2 - 60, h * 0.4], [w // 2 + 60, h * 0.4], [w * 0.01, h*0.9], [w*0.99, h*0.9]])
    destination = np.float32([[0, 0], [w-15, 0], [0, h], [w-15, h]])
    
    
    
    
    
    ################################################################################
    
    # 이미지 처리(그림은 나중에 한번에 그림)
    box_candis = ImageProcess.TS_Candidates(img)
    
    boxes, class_ids, confidences = ImageProcess.YOLO_JUDGE_DT(img, box_candis)
    
    X1, Y1, X2, Y2 = ImageProcess.StopLine(img)
    
    thresh, minverse, draw_info = ImageProcess.LANEBYHLS(img, source, destination)
    
    ################################################################################



    ################################################################################
    # 원본 위에 그림(상자)그리기(단계별로 yolo->sl->lane_hls)
    
    img_yolo = ImageProcess.YOLO_Labeling(img, boxes, class_ids, confidences)

    img_yolo_sl = ImageProcess.DrawStopLine(img_yolo, X1, Y1, X2, Y2)

    meanPts, result = ImageProcess.drawLANEBYHLS(img_yolo_sl, thresh, minverse, draw_info)
    
    ################################################################################

    result = ImageProcess.drawCandidates(result, box_candis)
    
    
    
    cv2.imshow("result", result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

cv2.destroyAllWindows()