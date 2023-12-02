#creating Hand Tracking Module so we can use it in other projects

import cv2
import mediapipe as mp
import time
import math

'''create a class called handDetector with two member functions in it,
named findHands and findPosition.
The function findHands will accept an RGB image and detects the hand in the frame and
locate the key points and draws the landmarks, the function findPosition
will give the position of the hand along with the id'''

class HandDetector():
    def __init__(self , mode = False , maxHands = 2 , modelC=1 , detectionCon = 0.5 , trackCon = 0.5) :  #here modelC = 1 is important
        self.mode = mode
        self.maxHands = maxHands
        self.modelC = modelC
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands ##detect the hands, in default ( look inside the class “Hands()“)
        self.hands = self.mpHands.Hands(self.mode , self.maxHands , self.modelC  , self.detectionCon , self.trackCon)  # this object only uses RGB

        self.mpDraw = mp.solutions.drawing_utils  #to draw the key points

        # self.lmlist = []

    def findHands(self , img , draw = True):
        imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)  #this is to verify whether the hand is detected or not

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img , handLms , self.mpHands.HAND_CONNECTIONS)
        return img


    def findPosition(self , img  , draw = True):
        h, w, c = img.shape
        self.lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks
            for handlms in myHand:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for id , lm in enumerate(handlms.landmark):
                    # print(id , lm)
                    cx , cy = int(lm.x*w) , int(lm.y*h)
                    # print(id , cx , cy)
                    self.lmlist.append([id, cx , cy])
                    # print(lmlist)
                    #for bounding box (checking max condition to create largest possible bounding box)
                    if cx > x_max:
                        x_max = cx
                    if cx < x_min:
                        x_min = cx
                    if cy > y_max:
                        y_max = cy
                    if cy < y_min:
                        y_min = cy

                if draw:
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    # cv2.circle(img, (cx, cy), 7, (0, 255, 255), -1)

        return self.lmlist

    def findDistance(self, p1  , p2 , img=None , draw = True):

            # x1, y1 = self.lmlist[p1][1], self.lmlist[p1][2]
            # x2, y2 = self.lmlist[p2][1], self.lmlist[p2][2]
            x1 = self.lmlist[p1][1]
            y1 = self.lmlist[p1][2]
            x2 = self.lmlist[p2][1]
            y2 = self.lmlist[p2][2]


            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            length = math.hypot(x2 - x1, y2 - y1)


            if draw:
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)


            return length, img, [x1, y1, x2, y2, cx, cy]

    def findAngle(self, img, p1, p2 ,pp1 , pp2 , draw=True):
        """
        Finds angle between three points. Inputs index values of landmarks
        instead of the actual points.
        :param img: Image to draw output on.
        :param p1: Point1 - Index of Landmark 1.
        :param p2: Point2 - Index of Landmark 2.
        :param pp1(base point): Point3 - Index of Landmark 3.
        :param pp2(base point): Point4 - Index of Landmark 4.
        :param draw:  Flag to draw the output on the image.
        :return:
        """
        # central point(c) joining (p1-c) and (p2-c)
        pp1x = self.lmlist[pp1][1]
        pp1y = self.lmlist[pp1][2]
        pp2x = self.lmlist[pp2][1]
        pp2y = self.lmlist[pp2][2]

        c1 , c2 = (pp1x+pp2x)//2 , (pp1y+pp2y)//2

        # Get the landmarks
        x1, y1 = self.lmlist[p1][1:]
        x2, y2 = self.lmlist[p2][1:]
        x3, y3 = c1 ,c2

        # Calculate the Angle ( make central point origin)
        angle = math.degrees(math.atan2(y2 - y3, x2 - x3) -
                             math.atan2(y1 - y3, x1 - x3))
        if angle < 0:
            angle += 180

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x3 ,y3), (0, 255, 0), 3)
            cv2.line(img, (x2, y2), (x3 ,y3), (0, 255, 0), 3)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            # cv2.putText(img, str(int(angle)), (x2 - 50, y2 -20),
            #             cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 5)
        return angle


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = cv2.flip(img , 1)
        img = detector.findHands(img)
        lmlist  = detector.findPosition(img)
        lmlist1 = detector.lmlist


        # if len(lmlist1) != 0:
        #     # print("empty")
        #     print(lmlist[4])

        if lmlist:
            ang = detector.findAngle(img, 8, 12, 5, 9, draw=True)
            print(ang)

        ######### To find FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (18, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (255, 0, 255), 3)
        #########

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.waitKey(0)
cv2.destroyAllWindows()


if __name__ == "__main__" :
    main()