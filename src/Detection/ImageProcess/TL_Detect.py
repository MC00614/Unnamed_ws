################### 만들어야할 알고리즘 ###################
# 컬러필터를 red light, yellow light, green light 각각을 값을 조절하여 해당 색만 나오게 하고, 거기에 candidatesBox 알고리즘 사용
# 각 색깔별로 변수를 만들어서 박스 생성 시 변수==1, 박스 없을 시 변수==0 으로 만듦
# 주의할 점 : 박스가 계속 생겻다가 사라졋다가 하기 때문에, 이를 보완할 방법을 생각해야 함.

import cv2
import numpy as np

initialized = False

# 직전 상태를 신호등 탐지 못한 상태와
# 카운트 = 0
tlnow = 0
tlcnt = 0

def nothing(a=1):
    return
    
def TL_Detect(img):
    # print('traffic_detect IN')
    global initialized, tlnow, tlcnt
    # global 
    W = img.shape[1]        # 이미지 가로 픽셀 개수
    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = traffic_roi(hsv,W)         # traffic_roi는 여기서 구현한 내장함수
    img = traffic_roi(img,W)

    lower_R = np.array([27, 0, 228])
    upper_R = np.array([140, 43, 255])
    lower_Y = np.array([50, 224, 243])
    upper_Y = np.array([145, 255, 255])
    lower_G = np.array([123, 234, 35])
    upper_G = np.array([215, 255, 199])

    mask_R = cv2.inRange(img, lower_R, upper_R)
    masked_R = cv2.bitwise_and(img, img, mask = mask_R)    # 빨간색으로 마스킹한 부분
    mask_Y = cv2.inRange(img, lower_Y, upper_Y)
    masked_Y = cv2.bitwise_and(img, img, mask = mask_Y)    # 노란색으로 마스킹한 부분
    mask_G = cv2.inRange(img, lower_G, upper_G)
    masked_G = cv2.bitwise_and(img, img, mask = mask_G)    # 초록색으로 마스킹한 부분
    
    # cv2.imshow('img1', masked_R)
    # cv2.imshow('img2', masked_Y)
    # cv2.imshow('img3', masked_G)
    # print('mask IN')

    box_red = BOX(masked_R)                                # 빨간색불의 정보 받아오기 
    box_yellow = BOX(masked_Y)                             # 노란색불의 정보 받아오기
    box_green = BOX(masked_G)                              # 초록색불의 정보 받아오기
    # print('box IN')

    # 직전 상태 / 임시 연속 카운트(0으로 세팅)
    # TLbox, TLnow, TLcnt = ImageProcess.TL_Detect(img, TLnow, TLcnt)

    # 색깔이 하나라도 들어오면 tlcnt 초기화

    # else빼고는 신호등 읽은 경우
    if len(box_red)>0:                                     # 빨간색이 들어왔을 때는 빨간색의 정보 보내기
        box = box_red
        tllog = 1
    elif len(box_yellow)>0:
        box = box_yellow
        tllog = 2
    elif len(box_green)>0:
        box = box_green
        tllog = 3
    else:
        # 지금 신호등 빛이 나오지 않았거나 탐지 못한경우
        box = []
        # 5번까지 못읽었으면 걍 신호등 벗어나거나 못찾은 것
        # tlcnt 너무 커질까봐 대충 10으로 줌
        # 폭탄 터지는 케이스
        if tlcnt >= 5:
            tlcnt = 10
            tlnow = 0
            return box, tlnow
        # 내 앞에 신호등이 켜져있지만 읽지 못한 경우가 포함
        # 신호등 조금 못읽었을 때는 cnt올리면서 폭탄 크기 키우기
        else:
            tlcnt += 1
            return box, tlnow

    # 내 앞에 있는 신호등이 하나라도 빛나는 경우에 아래의 코드로 실행
    # tllog 즉 지금 들어온 값이 직전상태와 다른 경우
    # 신호등 잡아내면 폭탄 0으로 초기화

    tlcnt = 0
    
    if tlnow != tllog:
        # 직전 상태를 현재 상태로 업데이트
        tlnow = tllog
        return box, tlnow

    # 현상유지
    # 현재 상태 == 현재값
    else:
        return box, tlnow


    # if len(box_red)>0:                                     # 빨간색이 들어왔을 때는 빨간색의 정보 보내기
    #     tlcnt += -1
    #     if tlcnt <= -2:
    #         return box_red, 1, tlcnt
    # elif len(box_yellow)>0:
    #     tlcnt = 0
    #     return box_yellow, 2, tlcnt
    # elif len(box_green)>0:
    #     tlcnt += 1
    #     if tlcnt >= 2:
    #         return  box_green, 3, tlcnt
    # else:
    #     stack += 1
    #     if stack >= 2:
    #         tlcnt = 0
    #         return [], 0, tlcnt

# def traffic_control(event):
#     if event=='red':
#         print('RED LIGHT!!')
#         # control : speed = 0
#     elif event=='yellow':
#         print('YELLOW LIGHT!!')
#         # control : speed : -=
#     elif event=='green':
#         print('GREEN LIGHT!!')
#         # control : speed = +
#     return None

# 전체 이미지에서 우반면만 체크
def traffic_roi(img,W):
    roi = img[:,W//2:,:]
    return roi

# 마스킹된 이미지를 받는다.
def BOX(img):
    box = []
    edges = cv2.Canny(img,40,120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)     # 외각 따기
    W = img.shape[1]                                           # 가로 길이
    try:
        for contr in contours:
            A = cv2.contourArea(contr,False)                   # 영역의 넓이
            # L = cv2.arcLength (contr, False)                 # 영역의 둘레
            x,y,w,h = cv2.boundingRect(contr)
            if A>30:
                if (w/h<1.4 and h/w<1.4):                      # 너무 비율이 박살난 직사각형이 아니면 출력
                    box.append([W+x,y,w,h])
        return box
    except:
        return box

# drawing part에서 호출한다.
##################### 박스 해놨당 #############################
def drawTraffic(img, box, color, allblack=True):
    if color == 0:
        return img
    boxcolor = (0,0,0)
    # color가 1, 2, 3이면 각각 빨 노 초를 뜻한다.
    if allblack==False:
        if color == 1:
            boxcolor = (0,0,255)
        elif color == 2:
            boxcolor = (0,255,255)
        elif color == 3:
            boxcolor = (0,255,0)
    for i in range(len(box)):
        x,y,w,h = box[i]
        cv2.rectangle(img, (x, y), (x+w, y+h), boxcolor, 10)
    return img

