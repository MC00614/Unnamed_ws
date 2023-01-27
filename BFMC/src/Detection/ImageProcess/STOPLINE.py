import cv2
import numpy as np
    
def StopLine(src):
    img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    img_canny = cv2.Canny(img_blur, 40, 120, apertureSize = 3, L2gradient = True)
    lines = cv2.HoughLinesP(img_canny, 0.8, np.pi / 180, 90, minLineLength = 20, maxLineGap = 60)

    # 조건
    # 1. 중앙을 지나는 직선
    # 2. 기울기가 5도 이하
    # 3. 가장 가까운 가로선
    # 4. 중앙 하단부분에만 포커스
    # + 차선 인식해서 끝부분과 맞닿게 하면 좋을것 같음
    # + HoughLine 파라미터 수정 필요할수도 (Canny Edge는 Candidate에서 이미 조정된 값)
    # + Canny를 괜히 2번하는중

    angle_treshold =  5 # in degree
    ath = np.tan(np.radians(angle_treshold))
    size = src.shape 
    midX = size[1]//2
    midY = size[0]*0.5
    X1, Y1, X2, Y2 = 0, 0, 0, 0
    
    try:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1-x2==0:
                 x1+=0.001
            if y2>midY and -ath<(y1-y2)/(x1-x2)<ath and (x1-midX)*(x2-midX)<=0 :
                if  y1>Y1:
                    X1, Y1, X2, Y2 = x1, y1, x2, y2
        return X1, Y1, X2, Y2
    except:
        return X1, Y1, X2, Y2

def DrawStopLine(src, X1, Y1, X2, Y2):
    if Y1:
        cv2.line(src, (X1, Y1), (X2, Y2), (255, 0, 0), 3)
    return src