import cv2

def ROI(img, scale=1, on=False):
    img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
    
    edges = cv2.Canny(img_blur,40,120)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    size_min = img.shape[0]

    box=[]
    for contr in contours:
        A = cv2.contourArea(contr,False)
        L = cv2.arcLength (contr, False)
        if A>size_min:
            x,y,w,h = cv2.boundingRect(contr)
            if 4*3.14*A/(L**2)>0.2:
                box.append((x, y, x+w, y+h))
                if on:
                    cv2.drawContours(img, contr, -1, (0,255,0), 3)
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
    if on:
        return img, box
    return box

def useWEBCAM():
    webcam = cv2.VideoCapture(0)

    while webcam.isOpened():
        status, frame = webcam.read()
        if status:
            img, cnt = ROI(frame, scale=0.5, on=True)
            cv2.imshow("영역", img)
            print(cnt)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return