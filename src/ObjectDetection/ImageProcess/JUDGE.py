def JUDGE(box1, box2, class_id, confidence):
    class_ids = []
    confidences = []
    boxes = []
    for candidate in range(len(pbox)):
        if 0.3 < IoU((x, y, x+w, x+h), pbox[candidate]):
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    return boxes, class_ids, confidences
