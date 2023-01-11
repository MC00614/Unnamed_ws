import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import ImageProcess
import CameraProcess

import numpy as np
import cv2
import time

cap = cv2.VideoCapture('./TESTDATA/bfmc2020_online_2.avi')
    
frame_cnt=0

################################################################################
# Video Save (동영상 저장)
# fps = 25.40
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')            # 인코딩 포맷 문자
# width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# size = (int(width), int (height))                   # 프레임 크기
# out = cv2.VideoWriter('record.mp4', fourcc, fps, size) # VideoWriter 객체 생성
# Video Save (동영상 저장)
#################################################################################

while True:
    ################################################################################
    # For VideoCapture (동영상 및 웹캠 사용시 주석 해제)
    retval, img = cap.read()
    if not retval : break
    # For VideoCapture (동영상 및 웹캠 사용시 주석 해제)
    scale = 0.5
    img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    ################################################################################

    ################################################################################
    # For L515 (사용시 주석 해제)
    img = CameraProcess.RGB_L515()
    # For L515 (사용시 주석 해제)
    ################################################################################

    ################################################################################
    # Stream Speed Control!
    # if frame_cnt<2: 
    #     frame_cnt+=1
    #     continue
    # else:
    #     # time.sleep(0.02)
    #     frame_cnt=0
    # Stream Speed Control!
    ################################################################################

    ################################################################################
    # 4PTs for Bird Eye View (BIV를 하려는 점 4개를 선택한 후 Enter)
    # 4 Enter if you don't want LANE (차선 인식 안쓸거면 엔터 4번 누르세요)
    source, destination = ImageProcess.PT4(img, trashframe=20, offset=50)
    if source[0][0] == False : continue
    ################################################################################
   


    ################################################################################
    # Image Process Part (그림은 나중에 한번에 그림)
    box_candis = ImageProcess.TS_Candidates(img)
    
    boxes, class_ids, confidences = ImageProcess.YOLO_JUDGE_DT(img, box_candis)
    
    SL, SLlocation = ImageProcess.StopLine(img)
    
    thresh, minverse, draw_info = ImageProcess.LANEBYHSV(img, source, destination)

    TLbox, TLcolor = ImageProcess.TL_Detect(img)
    # Image Process Part (그림은 나중에 한번에 그림)
    ################################################################################



    ################################################################################
    # Drawing Part (yolo->lane_hls->sl) (원본 위에 그림 그리기)
    img_yolo = ImageProcess.YOLO_Labeling(img, boxes, class_ids, confidences)
        
    meanPts, img_yolo_lane = ImageProcess.drawLANEBYHSV(img_yolo, thresh, minverse, draw_info)

    img_yolo_lane_sl = ImageProcess.DrawStopLine(img_yolo_lane, SL, meanPts)

    result = ImageProcess.drawTraffic(img_yolo_lane_sl, TLbox, TLcolor, allblack=True)

    result = ImageProcess.drawCandidates(result, box_candis)
    # Drawing Part (yolo->lane_hls->sl) (원본 위에 그림 그리기)
    ################################################################################

    

    ################################################################################
    # Information Print Part (처리된 값 출력하기)
    print('\n')
    print(f'Lane = {meanPts}')
    print(f'Stop Line = {SLlocation}')
    print(f'Object = {class_ids}')
    print(f'Traffic Sign = {TLcolor}')
    # Information Print Part (처리된 값 출력하기)
    ################################################################################


    ################################################################################
    # Video Save (동영상 저장)
    # out.write(result)
    # Video Save (동영상 저장)
    ################################################################################



    cv2.imshow("result", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break


# Video Save (동영상 저장)
# cap.release()
# Video Save (동영상 저장)

cv2.destroyAllWindows()
