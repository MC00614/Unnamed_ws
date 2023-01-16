import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import ImageProcess
import CameraProcess
import AfterProcess

import numpy as np
import cv2
import time

cap = cv2.VideoCapture('./TESTDATA/bfmc2020_online_2.avi')

# 동영상 크기배율 설정 (0<scale<1)
scale = 0.5
frame_cnt=0

################################################################################
# Video Save (동영상 저장 - q를 눌러서 종료하거나 전체 비디오 기다려야 저장)
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)*scale
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*scale
size = (int(width), int (height))
name = time.strftime('%Y.%m.%d-%H.%M.%S')
name = './Record/' + str(name) + '.mp4'
out = cv2.VideoWriter(name, fourcc, fps, size) # VideoWriter 객체 생성
# Video Save (동영상 저장 - q를 눌러서 종료하거나 전체 비디오 기다려야 저장)
#################################################################################

while True:
    ################################################################################
    # For VideoCapture (동영상 및 웹캠 사용시 주석 해제)
    retval, img = cap.read()
    if not retval : break
    img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # For VideoCapture (동영상 및 웹캠 사용시 주석 해제)
    ################################################################################

    ################################################################################
    # For L515 (사용시 주석 해제)
    # img = CameraProcess.RGB_L515()
    # For L515 (사용시 주석 해제)
    ################################################################################

    ################################################################################
    # Stream Speed Control!
    # if frame_cnt<3: 
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
    source, destination = ImageProcess.PT4(img, trashframe=20, offset=100)
    if source[0][0] == False : continue
    ################################################################################
   


    ################################################################################
    # Image Process Part (그림은 나중에 한번에 그림)
    SL, SLlocation = ImageProcess.StopLine(img)
    
    thresh, minverse, draw_info = ImageProcess.LANEBYHSV(img, source, destination)

    # 직전 상태 / 카운트
    TLbox, TLnow = ImageProcess.TL_Detect(img)

    box_candis = ImageProcess.TS_Candidates(img)
    
    boxes, class_ids, confidences = ImageProcess.YOLO_JUDGE_DT(img, box_candis, SLlocation)
    # Image Process Part (그림은 나중에 한번에 그림)
    ################################################################################


    ################################################################################
    # Drawing Part (yolo->lane_hls->sl) (원본 위에 그림 그리기)
    img_yolo = ImageProcess.YOLO_Labeling(img, boxes, class_ids, confidences)
        
    meanPts, img_yolo_lane = ImageProcess.drawLANEBYHSV(img_yolo, thresh, minverse, draw_info)

    img_yolo_lane_sl = ImageProcess.DrawStopLine(img_yolo_lane, SL, meanPts)

    result = ImageProcess.drawTraffic(img_yolo_lane_sl, TLbox, TLnow, allblack=False)

    # result = ImageProcess.drawCandidates(result, box_candis)
    # Drawing Part (yolo->lane_hls->sl) (원본 위에 그림 그리기)
    ################################################################################

    

    ################################################################################
    # Information Print Part (처리된 값 출력하기)
    print('\n')
    print(f'Lane = {meanPts}')
    print(f'Stop Line = {SLlocation}')
    print(f'Object = {class_ids}')
    print(f'Traffic Sign = {TLnow}')
    # Information Print Part (처리된 값 출력하기)
    ################################################################################
    

    ################################################################################
    # Put Text in Video (동영상에 글씨 쓰기)
    text1 = meanPts
    result = AfterProcess.PutTEXT(result, text1, 0.5, 0.99, color = (255,255,255))
    text2 = img.shape[0]-SLlocation
    if text2<600 and meanPts[0]!=False and meanPts[1]!=False:
        result = AfterProcess.PutTEXT(result, text2, 0.2, 0.99, color = (0,0,0))
    if TLnow==1:
        text3 = 'Red'
        result = AfterProcess.PutTEXT(result, text3, 0.8, 0.99, color = (0,0,255))
    elif TLnow==2 and text2<600:
        text3 = 'Yellow'
        result = AfterProcess.PutTEXT(result, text3, 0.8, 0.99, color = (0,255,255))
    elif TLnow==3:
        text3 = 'Green'
        result = AfterProcess.PutTEXT(result, text3, 0.8, 0.99, color = (0,255,0))
    # Put Text in Video (동영상에 글씨 쓰기)
    ################################################################################


    ################################################################################
    # Video Save (동영상 저장)
    out.write(result)
    # Video Save (동영상 저장)
    ################################################################################
    

    cv2.imshow("result", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Video Save (동영상 저장)
cap.release()
out.release()
# Video Save (동영상 저장)

cv2.destroyAllWindows()
