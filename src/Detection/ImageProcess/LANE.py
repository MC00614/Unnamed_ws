import cv2
import numpy as np

initialized = False

def nothing(a=1):
    return

def wrapping(image, source, destination):
    ################################################################################
    global initialized
    if initialized == False:       # 아래의 코드는 처음 실행했을 때, 한번만 실행되어야하므로 self.initialized를 사용해서 처음에만 실행되도록 짜준 코드
            cv2.namedWindow("LANE", cv2.WINDOW_NORMAL)       # Simulator_Image를 원하는 크기로 조정가능하도록 만드는 코드
            # 흰색 차선을 검출할 때, 이미지를 보면서 트랙바를 움직이면서 흰색선이 검출되는 정도를 파악하고 코드안에 그 수치를 넣어준다.
            cv2.createTrackbar('low_H', 'LANE', 0, 255, nothing)    # Trackbar 만들기
            cv2.createTrackbar('low_S', 'LANE', 0, 255, nothing)
            cv2.createTrackbar('low_V', 'LANE', 155, 255, nothing)
            cv2.createTrackbar('high_H', 'LANE', 255, 255, nothing)    
            cv2.createTrackbar('high_S', 'LANE', 255, 255, nothing)
            cv2.createTrackbar('high_V', 'LANE', 255, 255, nothing)
            initialized = True  # 두 번 다시 여기 들어오지 않도록 처리
    # ################################################################################
    (h, w) = (image.shape[0], image.shape[1])
    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    minv = cv2.getPerspectiveTransform(destination, source)
    _image = cv2.warpPerspective(image, transform_matrix, (w, h))
    return _image, minv

def color_filter(image, isyellow=False):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    low_H = cv2.getTrackbarPos('low_H', 'LANE')
    low_S = cv2.getTrackbarPos('low_S', 'LANE')
    low_V = cv2.getTrackbarPos('low_V', 'LANE')
    high_H = cv2.getTrackbarPos('high_H', 'LANE')
    high_S = cv2.getTrackbarPos('high_S', 'LANE')
    high_V = cv2.getTrackbarPos('high_V', 'LANE')

    lower = np.array([low_H, low_S, low_V])
    upper = np.array([high_H, high_S, high_V])

    if isyellow:
        yellow_lower = np.array([0, 85, 81])
        yellow_upper = np.array([190, 255, 255])

        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        white_mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_or(yellow_mask, white_mask)
        masked = cv2.bitwise_and(image, image, mask = mask)

        return masked
    else:
        mask = cv2.inRange(hsv, lower, upper)
        masked = cv2.bitwise_and(image, image, mask = mask)
        cv2.imshow("LANE", masked)
        return masked

def roi(image):
    x = int(image.shape[1])
    y = int(image.shape[0])

    # 한 붓 그리기
    _shape = np.array(
        [[int(0.1*x), int(y)], [int(0.1*x), int(0.1*y)], [int(0.4*x), int(0.1*y)], [int(0.4*x), int(y)], [int(0.7*x), int(y)], [int(0.7*x), int(0.1*y)],[int(0.9*x), int(0.1*y)], [int(0.9*x), int(y)], [int(0.2*x), int(y)]])

    mask = np.zeros_like(image)

    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, np.int32([_shape]), ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def plothistogram(image):
    histogram = np.sum(image[image.shape[0]//2:, :], axis=0)
    midpoint = np.int64(histogram.shape[0]/2)
    leftbase = np.argmax(histogram[:(int(midpoint))])
    rightbase = np.argmax(histogram[int(midpoint):]) + int(midpoint)
    return leftbase, rightbase

def slide_window_search(binary_warped, left_current, right_current):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    nwindows = 4
    window_height = np.int64(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()  # 선이 있는 부분의 인덱스만 저장
    nonzero_y = np.array(nonzero[0])  # 선이 있는 부분 y의 인덱스 값
    nonzero_x = np.array(nonzero[1])  # 선이 있는 부분 x의 인덱스 값
    margin = 20
    minpix = 100
    left_lane = []
    right_lane = []
    color = [0, 255, 0]
    thickness = 2

    for w in range(nwindows):
        win_y_low = binary_warped.shape[0] - (w + 1) * window_height  # window 윗부분
        win_y_high = binary_warped.shape[0] - w * window_height  # window 아랫 부분
        win_xleft_low = left_current - margin  # 왼쪽 window 왼쪽 위
        win_xleft_high = left_current + margin  # 왼쪽 window 오른쪽 아래
        win_xright_low = right_current - margin  # 오른쪽 window 왼쪽 위
        win_xright_high = right_current + margin  # 오른쪽 window 오른쪽 아래

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)
        good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
        left_lane.append(good_left)
        right_lane.append(good_right)
        # cv2.imshow("Slide Window Search", out_img)

        if len(good_left) > minpix:
            left_current = np.int64(np.mean(nonzero_x[good_left]))
        if len(good_right) > minpix:
            right_current = np.int64(np.mean(nonzero_x[good_right]))

    left_lane = np.concatenate(left_lane)  # np.concatenate() -> array를 1차원으로 합침
    right_lane = np.concatenate(right_lane)

    leftx = nonzero_x[left_lane]
    lefty = nonzero_y[left_lane]
    rightx = nonzero_x[right_lane]
    righty = nonzero_y[right_lane]
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    try :
        left_fit = np.polyfit(lefty, leftx, 2)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        ltx = np.trunc(left_fitx)  # np.trunc() -> 소수점 부분을 버림
        out_img[nonzero_y[left_lane], nonzero_x[left_lane]] = [255, 0, 0]
    except :
        ltx = [False]
    
    try :
        right_fit = np.polyfit(righty, rightx, 2)
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        rtx = np.trunc(right_fitx)
        out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = [0, 0, 255]
    except :
        rtx = [False]
    ret = {'left_fitx' : ltx, 'right_fitx': rtx, 'ploty': ploty}
    return ret


def draw_lane_lines(original_image, warped_image, Minv, draw_info):
    left_fitx = draw_info['left_fitx']
    right_fitx = draw_info['right_fitx']
    ploty = draw_info['ploty']
    
    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    if left_fitx[0]==False:
        if right_fitx[0]==False:
            LRpts = [False, False]
            return LRpts, original_image

        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        LRpts = [False, round(np.mean(pts_right[:10]))]

        cv2.line(color_warp, (int(right_fitx[-1]),len(right_fitx)), (int(right_fitx[0]),0), (0,0,255), 4)
        newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
        result = cv2.addWeighted(original_image, 1, newwarp, 1, 0)

        return LRpts, result

    if right_fitx[0]==False:
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        LRpts = [round(np.mean(pts_left[:10])), False]

        cv2.line(color_warp, (int(left_fitx[-1]),len(left_fitx)), (int(left_fitx[0]),0), (0,0,255), 4)
        newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))

        result = cv2.addWeighted(original_image, 1, newwarp, 1, 0)
        return LRpts, result

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    mean_x = np.mean((left_fitx, right_fitx), axis=0)

    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

    cv2.fillPoly(color_warp, np.int_([pts]), (216, 168, 74))
    cv2.fillPoly(color_warp, np.int_([pts_mean]), (216, 168, 74))

    newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
    result = cv2.addWeighted(original_image, 1, newwarp, 0.4, 0)

    return [round(np.mean(pts_left)), round(np.mean(pts_right))], result

def LANEBYHSV(img, source, destination):
    wrapped_img, minverse = wrapping(img, source, destination)
    w_f_img = color_filter(wrapped_img)
    w_f_r_img = roi(w_f_img)
    _gray = cv2.cvtColor(w_f_r_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(_gray, 160, 255, cv2.THRESH_BINARY)
    leftbase, rightbase = plothistogram(thresh)
    draw_info = slide_window_search(thresh, leftbase, rightbase)
    return thresh, minverse, draw_info

def drawLANEBYHSV(img, thresh, minverse, draw_info):
    meanPts, img = draw_lane_lines(img, thresh, minverse, draw_info)
    if meanPts[0]<0 or img.shape[1]<meanPts[0]:
        meanPts[0] = False
    if meanPts[1]<0 or img.shape[1]<meanPts[1]:
        meanPts[1] = False
    return meanPts, img