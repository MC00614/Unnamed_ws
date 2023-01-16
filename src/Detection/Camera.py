import CameraProcess
from sensor_msgs.msg import CompressedImage


import rospy
import cv2
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Int32, Int32MultiArray

class CameraReceiver():
    def __init__(self):
        # rospy.loginfo("Camera Receiver object is created")
        # 토픽 수신
        rospy.Subscriber("/automobile/image_raw/compressed", CompressedImage, self.camera_callback)
        self.bridge = CvBridge()
        # 토픽 발신
        self.SIZE = rospy.Publisher("/automobile/Detection/SIZE", Int32MultiArray, queue_size=1)
        self.LANE = rospy.Publisher("/automobile/Detection/LANE", Int32MultiArray, queue_size=1)
        self.OBJECT = rospy.Publisher("/automobile/Detection/OBJECT", Int32MultiArray, queue_size=1)
        self.SL = rospy.Publisher("/automobile/Detection/SL", Int32, queue_size=1)

    # # ==================================================================================
    def camera_callback(self, data):
        img = self.bridge.compressed_imgmsg_to_cv2(data)
        ################################################################################
        # 4PTs for Bird Eye View (BIV를 하려는 점 4개를 선택한 후 Enter)
        # 4 Enter if you don't want LANE (차선 인식 안쓸거면 엔터 4번 누르세요)
        source, destination = ImageProcess.PT4(img, trashframe=20, offset=100)
        if source[0][0] == False : return
        ################################################################################
    


        ################################################################################
        # Image Process Part (그림은 나중에 한번에 그림)
        SL, SLlocation = ImageProcess.StopLine(img)
        
        thresh, minverse, draw_info = ImageProcess.LANEBYHSV(img, source, destination)

        # 직전 상태 / 카운트
        TLbox, TLnow = ImageProcess.TL_Detect(img)

        box_candis = ImageProcess.TS_Candidates(img)
        
        boxes, class_ids, confidences = ImageProcess.YOLO_JUDGE_DT(img, box_candis, SLlocation)
        # Image Process Part (그림은 나중에 한번에 그림)
        ################################################################################


        ################################################################################
        # Drawing Part (yolo->lane_hls->sl) (원본 위에 그림 그리기)
        img_yolo = ImageProcess.YOLO_Labeling(img, boxes, class_ids, confidences)
            
        meanPts, img_yolo_lane = ImageProcess.drawLANEBYHSV(img_yolo, thresh, minverse, draw_info)

        img_yolo_lane_sl = ImageProcess.DrawStopLine(img_yolo_lane, SL, meanPts)

        result = ImageProcess.drawTraffic(img_yolo_lane_sl, TLbox, TLnow, allblack=False)

        # result = ImageProcess.drawCandidates(result, box_candis)
        # Drawing Part (yolo->lane_hls->sl) (원본 위에 그림 그리기)
        ################################################################################

        

        ################################################################################
        # Information Print Part (처리된 값 출력하기)
        print('\n')
        print(f'Lane = {meanPts}')
        print(f'Stop Line = {SLlocation}')
        print(f'Object = {class_ids}')
        print(f'Traffic Sign = {TLnow}')
        # Information Print Part (처리된 값 출력하기)
        ################################################################################
        

        ################################################################################
        # Put Text in Video (동영상에 글씨 쓰기)
        text1 = meanPts
        result = AfterProcess.PutTEXT(result, text1, 0.5, 0.99, color = (255,255,255))
        text2 = img.shape[0]-SLlocation
        if text2<600 and meanPts[0]!=False and meanPts[1]!=False:
            result = AfterProcess.PutTEXT(result, text2, 0.2, 0.99, color = (0,0,0))
        if TLnow==1:
            text3 = 'Red'
            result = AfterProcess.PutTEXT(result, text3, 0.8, 0.99, color = (0,0,255))
        elif TLnow==2 and text2<600:
            text3 = 'Yellow'
            result = AfterProcess.PutTEXT(result, text3, 0.8, 0.99, color = (0,255,255))
        elif TLnow==3:
            text3 = 'Green'
            result = AfterProcess.PutTEXT(result, text3, 0.8, 0.99, color = (0,255,0))
        # Put Text in Video (동영상에 글씨 쓰기)
        ################################################################################

        # 중점을 토픽으로 쏴준다.
        self.SIZE.publish(Int32MultiArray(img.shape))
        self.LANE.publish(Int32MultiArray(meanPts))
        self.OBJECT.publish(Int32MultiArray(class_ids))
        if meanPts[0]!=False and meanPts[1]!=False:
            self.SL.publish(Int32MultiArray(SL))


        cv2.imshow("result", result)

        cv2.waitKey(1)

def nothing():
    pass

# 실제 코드 동작 시, 실행할 코드
def run():
    rospy.init_node("Detection")     # camera_example이라는 이름으로 노드를 ROS_MASTER에 등록해주는 코드 (이름이 겹치지만 않으면 됨) 
    Camera()          # 실제 동작을 담당할 Object
    rospy.spin()                         # ROS Node가 종료되지 않고, Callback 함수를 정상적으로 실행해주기 위한 부분 (코드를 유지하면서, 지속적으로 subscriber를 호출해주는 함수)

if __name__ == '__main__':
    run()
