import cv2
import numpy as np

def PT4(img, trashframe=5, offset=50):
    global XY
    global source, destination
    try:
        if len(XY)<trashframe:
            XY.append([False,False])
            return [[False]], False
        elif len(XY)<trashframe+4:
            x, y, _, _ = cv2.selectROI('Pick 4Pts for BIV',img)
            XY.append((x+1,y+1))
            if len(XY)==(trashframe+4):
                (h, w) = (img.shape[0], img.shape[1])
                source = np.float32(XY[-4:])
                destination = np.float32([[offset, offset], [offset, h-offset], [w-offset, offset], [w-offset, h-offset]])
                cv2.destroyAllWindows()
                return source, destination
            return [[False]], False
        return source, destination
    except:
        XY=[[False,False]]
        return [[False]], False