import cv2
def PutTEXT(img, text, X, Y, color = (255,255,255)):
    text = str(text)
    font = cv2.LINE_AA
    H, W = img.shape[0], img.shape[1]
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    textX = (W - textsize[0])*X
    textY = (H - textsize[1])*Y
    img = cv2.putText(img, text, (int(textX), int(textY)), font, 1, color, 2)
    return img