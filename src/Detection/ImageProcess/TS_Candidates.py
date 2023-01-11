import cv2

def TS_Candidates(img, scale=1, on=False):
    img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
    
    edges = cv2.Canny(img_blur,40,120)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (H,W) = (img.shape[0], img.shape[1])
    box=[]

    # 조건
    # 1. 넓이/길이제곱 비율
    # 2. 가로세로 비율
    # 3. 최소 넓이
    # 4. ROI 설정(현재는 우측2/3, 위쪽1/2)
    # + ROI설정 먼저하고 이미지 처리하는 방식으로 바꿔야함
    # + hsv는 사용 안함

    for contr in contours:
        A = cv2.contourArea(contr,False)
        L = cv2.arcLength (contr, False)
        if A>(W//2):
            x,y,w,h = cv2.boundingRect(contr)
            if 4*3.14*A/(L**2)>0.3 and (w/h<1.9 and h/w<1.9) and x>(W*0.5) and y<(H*0.5):
                box.append((x, y, x+w, y+h))
                if on:
                    cv2.drawContours(img, contr, -1, (0,255,0), 3)
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
    if on:
        return img, box
    return box

def drawCandidates(img, box):
    for i in range(len(box)):
        x, y, w, h = box[i]
        cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 3)
    return img

def IFuseWEBCAM():
    webcam = cv2.VideoCapture(0)

    while webcam.isOpened():
        status, frame = webcam.read()
        if status:
            img, cnt = TS_Candidates(frame, scale=0.5, on=True)
            cv2.imshow("영역", img)
            print(cnt)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return