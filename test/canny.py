import numpy as np
import cv2
import os

img = cv2.imread('/images/123.jpg')
# img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
cv2.imshow("blur", img_blur)

edges = cv2.Canny(img_blur,40,120)
cv2.imshow("canny", edges)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
size_min = img.shape[0]



# mode – contours를 찾는 방법
# cv2.RETR_EXTERNAL : contours line중 가장 바같쪽 Line만 찾음.
# cv2.RETR_LIST : 모든 contours line을 찾지만, hierachy 관계를 구성하지 않음.
# cv2.RETR_CCOMP : 모든 contours line을 찾으며, hieracy관계는 2-level로 구성함.
# cv2.RETR_TREE : 모든 contours line을 찾으며, 모든 hieracy관계를 구성함.

# method – contours를 찾을 때 사용하는 근사치 방법
# cv2.CHAIN_APPROX_NONE : 모든 contours point를 저장.
# cv2.CHAIN_APPROX_SIMPLE : contours line을 그릴 수 있는 point 만 저장. (ex; 사각형이면 4개 point)
# cv2.CHAIN_APPROX_TC89_L1 : contours point를 찾는 algorithm
# cv2.CHAIN_APPROX_TC89_KCOS : contours point를 찾는 algorithm


i=0
pi = 3.14
for contr in contours:
    A = cv2.contourArea(contr,False)
    if A>size_min:
        x,y,w,h = cv2.boundingRect(contr)
        L = cv2.arcLength (contr, False)
        if 4*pi*A/(L**2)>0.3:
            # 이미 그린 사각형 근처에는 그리지 말아야함!!!
            # 내일 이거 해보자~~
            cv2.drawContours(img, contr, -1, (0,255,0), 3)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
            i+=1

cv2.imshow("영역", img)

print(i)

cv2.waitKey(0)
cv2.destroyAllWindows()
