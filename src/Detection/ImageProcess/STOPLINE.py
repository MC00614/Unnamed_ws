import cv2
import numpy as np


def nothing(a=1):
    return

initialized = False

def StopLine(src):
    ################################################################################
    global initialized
    if initialized == False:       # 아래의 코드는 처음 실행했을 때, 한번만 실행되어야하므로 self.initialized를 사용해서 처음에만 실행되도록 짜준 코드
        cv2.namedWindow("LANE", cv2.WINDOW_NORMAL)       # Simulator_Image를 원하는 크기로 조정가능하도록 만드는 코드
        # 흰색 차선을 검출할 때, 이미지를 보면서 트랙바를 움직이면서 흰색선이 검출되는 정도를 파악하고 코드안에 그 수치를 넣어준다.
        cv2.createTrackbar('low_H', 'LANE', 0, 255, nothing)    # Trackbar 만들기
        cv2.createTrackbar('low_S', 'LANE', 0, 255, nothing)
        cv2.createTrackbar('low_V', 'LANE', 230, 255, nothing)
        cv2.createTrackbar('high_H', 'LANE', 255, 255, nothing)    
        cv2.createTrackbar('high_S', 'LANE', 255, 255, nothing)
        cv2.createTrackbar('high_V', 'LANE', 255, 255, nothing)
        initialized = True  # 두 번 다시 여기 들어오지 않도록 처리
    #################################################################################

    # img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    low_H = cv2.getTrackbarPos('low_H', 'LANE')
    low_S = cv2.getTrackbarPos('low_S', 'LANE')
    low_V = cv2.getTrackbarPos('low_V', 'LANE')
    high_H = cv2.getTrackbarPos('high_H', 'LANE')
    high_S = cv2.getTrackbarPos('high_S', 'LANE')
    high_V = cv2.getTrackbarPos('high_V', 'LANE')

    lower = np.array([low_H, low_S, low_V])
    upper = np.array([high_H, high_S, high_V])

    mask = cv2.inRange(hsv, lower, upper)
    masked = cv2.bitwise_and(src, src, mask = mask)

    img_canny = cv2.Canny(masked, 40, 120, apertureSize = 3, L2gradient = True)

    W = src.shape[1]
    lines = cv2.HoughLinesP(img_canny, 1, np.pi / 180, 90, minLineLength = W*0.1, maxLineGap = 20)

    # 조건
    # 1. 중앙을 지나는 직선
    # 2. 기울기가 5도 이하
    # 3. 가장 가까운 가로선
    # 4. 중앙 하단부분에만 포커스
    # + 차선 인식해서 끝부분과 맞닿게 하면 좋을것 같음
    # + HoughLine 파라미터 수정 필요할수도 (Canny Edge는 Candidate에서 이미 조정된 값)
    # + Canny를 괜히 2번하는중

    angle_treshold =  3 # in degree
    ath = np.tan(np.radians(angle_treshold))
    size = src.shape 
    midX = size[1]//2
    midY = size[0]*0.5

    try:
        X1, Y1, X2, Y2 = 0, 0, 0, 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1-x2==0:
                 x1+=0.001
            if y2>midY and -ath<(y1-y2)/(x1-x2)<ath and (x1-midX)*(x2-midX)<=0 :
                if  y1>Y1:
                    X1, Y1, X2, Y2 = x1, y1, x2, y2                    
        sl = [X1, Y1, X2, Y2]
        location = (Y1+Y2)//2
        if location == 0:
            sl = False
            location = False
        return sl, location
    except:
        sl = False
        location = False
        return sl, location

def DrawStopLine(src, sl, meanPts):
    if sl==False or (meanPts[0]==False and meanPts[1]==False):
        return src
    X1, Y1, X2, Y2 = sl
    cv2.line(src, (X1, Y1), (X2, Y1), (0, 0, 0), 3)
    return src