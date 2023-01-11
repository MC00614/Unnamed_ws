import cv2
import numpy as np
import os

net = cv2.dnn.readNet("./ImageProcess/YOLODATA/BOSCHDATA/yolov3-tiny_10000-bfmc.weights", "./ImageProcess/YOLODATA/BOSCHDATA/yolov3-tiny-bfmc.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

classes = []
with open("./ImageProcess/YOLODATA/BOSCHDATA/obj-bfmc.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


def YOLO_JUDGE_DT(img, roi_box):

    box = []
    class_ids = []
    confidences = []
    
    # cadiBox없으면 욜로 안돌림
    if len(roi_box)==0:
        return box, class_ids, confidences
    
    (height, width) = (img.shape[0], img.shape[1])
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)


    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                if JUDGE(roi_box,(x, y, x+w, y+h)):
                    box.append((x, y, w, h))
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    return box, class_ids, confidences




def JUDGE(box_roi, box_yolo):
    for candidate in range(len(box_roi)):
        if 0.3 < IoU(box_yolo, box_roi[candidate]):
            return True
    return False




def IoU(box1, box2):
    # box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / box2_area
    return iou



def YOLO_Labeling(img, boxes, class_ids, confidences, classes=classes, colors=colors):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y-5), font, 1, color, 2)
    return img
