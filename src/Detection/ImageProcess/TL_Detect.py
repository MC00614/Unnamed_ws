################### 만들어야할 알고리즘 ###################
# 컬러필터를 red light, yellow light, green light 각각을 값을 조절하여 해당 색만 나오게 하고, 거기에 candidatesBox 알고리즘 사용
# 각 색깔별로 변수를 만들어서 박스 생성 시 변수==1, 박스 없을 시 변수==0 으로 만듦
# 주의할 점 : 박스가 계속 생겻다가 사라졋다가 하기 때문에, 이를 보완할 방법을 생각해야 함.

import cv2
import numpy as np

initialized = False


def nothing(a=1):
    return
    
def TL_Detect(img):
    # print('traffic_detect IN')
    global initialized
    W = img.shape[1]        # 이미지 가로 픽셀 개수
    
    # cv2.namedWindow("roi_Image", cv2.WINDOW_NORMAL)       # Simulator_Image를 원하는 크기로 조정가능하도록 만드는 코드
    # 흰색 차선을 검출할 때, 이미지를 보면서 트랙바를 움직이면서 흰색선이 검출되는 정도를 파악하고 코드안에 그 수치를 넣어준다.
    # if initialized==False:
        # cv2.createTrackbar('low_H', 'roi_Image', 0, 255, nothing)    # Trackbar 만들기
        # cv2.createTrackbar('low_S', 'roi_Image', 0, 255, nothing)
        # cv2.createTrackbar('low_V', 'roi_Image', 0, 255, nothing)
        # cv2.createTrackbar('high_H', 'roi_Image', 255, 255, nothing)    
        # cv2.createTrackbar('high_S', 'roi_Image', 255, 255, nothing)
        # cv2.createTrackbar('high_V', 'roi_Image', 255, 255, nothing)
        # initialized=True
    # low_H = cv2.getTrackbarPos('low_H', 'roi_Image')
    # low_S = cv2.getTrackbarPos('low_S', 'roi_Image')
    # low_V = cv2.getTrackbarPos('low_V', 'roi_Image')
    # high_H = cv2.getTrackbarPos('high_H', 'roi_Image')
    # high_S = cv2.getTrackbarPos('high_S', 'roi_Image')
    # high_V = cv2.getTrackbarPos('high_V', 'roi_Image')

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = traffic_roi(hsv,W)
    img = traffic_roi(img,W)

    # cv2.setTrackbarPos('low_H', 'roi_Image', 50)    # Trackbar 만들기
    # cv2.setTrackbarPos('low_S', 'roi_Image', 100)
    # cv2.setTrackbarPos('low_V', 'roi_Image', 200)
    # cv2.setTrackbarPos('high_H', 'roi_Image', 130)
    # cv2.setTrackbarPos('high_S', 'roi_Image', 255)
    # cv2.setTrackbarPos('high_V', 'roi_Image', 255)

    lower_R = np.array([27, 0, 228])
    upper_R = np.array([140, 43, 255])
    lower_Y = np.array([50, 224, 243])
    upper_Y = np.array([145, 255, 255])
    lower_G = np.array([123, 234, 35])
    upper_G = np.array([215, 255, 199])

    mask_R = cv2.inRange(img, lower_R, upper_R)
    masked_R = cv2.bitwise_and(img, img, mask = mask_R)
    mask_Y = cv2.inRange(img, lower_Y, upper_Y)
    masked_Y = cv2.bitwise_and(img, img, mask = mask_Y)
    mask_G = cv2.inRange(img, lower_G, upper_G)
    masked_G = cv2.bitwise_and(img, img, mask = mask_G)
    
    # cv2.imshow('img1', masked_R)
    # cv2.imshow('img2', masked_Y)
    # cv2.imshow('img3', masked_G)
    # print('mask IN')

    box_red = BOX(masked_R)
    box_yellow = BOX(masked_Y)
    box_green = BOX(masked_G)
    # print('box IN')

    if len(box_red)>0:
        return box_red, 1
    elif len(box_yellow)>0:
        return box_yellow, 2
    elif len(box_green)>0:
        return  box_green, 3
    else:
        return [], 0

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

def traffic_roi(img,W):
    roi = img[:,W//2:,:]
    return roi

def BOX(img):
    box = []
    edges = cv2.Canny(img,40,120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    W = img.shape[1]
    try:
        for contr in contours:
            A = cv2.contourArea(contr,False)
            # L = cv2.arcLength (contr, False)
            x,y,w,h = cv2.boundingRect(contr)
            if A>20:
                if (w/h<1.4 and h/w<1.4):
                    box.append([W+x,y,w,h])
        return box
    except:
        return box

##################### 박스 해놨당 #############################
def drawTraffic(img, box, color, allblack=True):
    if color == 0:
        return img
    boxcolor = (0,0,0)
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

