import ImageProcess
import cv2
import numpy as np
import copy

cap = cv2.VideoCapture(0)

frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

net = cv2.dnn.readNet("st2.weights", "yolov3-tiny.cfg")

classes = []
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


while True:
    retval, img = cap.read()
    if not retval:
        break

    # 민찬이가 만든 roi박스가 욜로이미지학습에 영향을 끼치지 않도록 깊은 복사로 완전히 새롭게 img_ROI 생성(img와는 완전 별개가 됨)
    img_ROI = copy.deepcopy(img)

    img_ROI, pbox = ImageProcess.ROI(img_ROI, scale=1)

    ImageProcess.YOLO_DT(img, pbox)

    cv2.imshow("result1", img_ROI)
    cv2.imshow("result2", img)

    key = cv2.waitKey(25)
    if key == 27:
        break

if cap.isOpened():
    cap.release()

cv2.destroyAllWindows()