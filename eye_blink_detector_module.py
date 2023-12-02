import cv2
import mediapipe as mp
import math
import time
from playsound import playsound

#to play sound
from pygame import mixer
mixer.init()
voice_click = mixer.Sound('sclick.mp3')

class FaceMeshDetector():

    def __init__(self, staticMode=False, maxFaces=2,refinelandmarks=True, minDetectionCon=0.5, minTrackCon=0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refinelandmarks = refinelandmarks
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces,self.refinelandmarks,
                                                 self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                           self.drawSpec, self.drawSpec)
                face = []
                for id,lm in enumerate(faceLms.landmark):
                    # print(id,lm)
                    ih, iw, ic = img.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                    #            0.7, (0, 255, 0), 1) #to show id of each point on img

                    # print(id,x,y)
                    # face.append([id,x,y])
                    face.append([x, y])
                faces.append(face)
        return img, faces


    def findDistance(self,p1, p2, img=None):

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length,info, img
        else:
            return length, info


    def EyeBlinkDetector(self,img, faces, draw=True):

        # left eye landmarks(refer mediapipe face landmark module)        ]
        idList_l = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]
        # right eye landmarks(refer mediapipe face landmark module)
        idList_r = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]

        # print(idList[0], idList[8], idList[12], idList[4])    #point of extremeties to calculate aspect ratio

        color = (255, 0, 255)
        if faces:
            face = faces[0]  # first face in faces list
            for id in idList_l:
                if draw:
                    cv2.circle(img, face[id], 1, color, cv2.FILLED)  # drawing eye landmarks
            for id in idList_r:
                if draw:
                    cv2.circle(img, face[id], 1, color, cv2.FILLED)  # drawing eye landmarks

        #parameters for left eye
        leftUp, leftDown  = face[159], face[145]
        leftLeft, leftRight = face[133], face[33]

        len_Ver_l, _ = FaceMeshDetector.findDistance(self,leftUp,leftDown)
        len_Hor_l, _ = FaceMeshDetector.findDistance(self,leftLeft, leftRight)

        #parameters for right eye
        rightUp, rightDown  = face[386], face[374]
        rightLeft, rightRight = face[263], face[362]

        len_Ver_r, _ = FaceMeshDetector.findDistance(self,rightUp,rightDown)
        len_Hor_r, _ = FaceMeshDetector.findDistance(self,rightLeft, rightRight)

        # if draw:
        #     cv2.line(img, leftUp, leftDown, (0,200,0), 1)
        #     cv2.line(img, leftLeft, leftRight, (0, 200, 0), 1)
        #
        #     cv2.line(img, rightUp, rightDown, (0,200,0), 1)
        #     cv2.line(img, rightLeft, rightRight, (0, 200, 0), 1)


        ratio_l = int((len_Ver_l/len_Hor_l)*100)  #to optimize against change in distance due to camera perspective
        ratio_r = int((len_Ver_r / len_Hor_r) * 100)
        # ratio = (ratio_l + ratio_r) / 2
        # print(ratio, ratio_r, ratio_l)

        if ratio_r < 23 and ratio_l < 23:
            blinked = True
            color = (0,200,0)
        else:
            blinked = False

        # cv2.putText(img, f'Blinked :{blinked}', (600, 120),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #                 fontScale=1, color=color, thickness=2)

        return blinked, img


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1080)
    cap.set(4, 720)
    pTime = 0
    detector = FaceMeshDetector(maxFaces=1)
    while True:
        success, img = cap.read()
        # img = cv2.flip(img , 1)
        # print(img.shape)

        img, faces = detector.findFaceMesh(img, False)
        if len(faces)!= 0:
            blink= detector.EyeBlinkDetector(img, faces, True)

        if faces:
            colour = (0, 0, 200)
            if blink:
                colour = (0, 200, 0)
            cv2.putText(img, f'Blinked :{blink}', (535, 140), fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=2, color=colour, thickness=2)
            # for left eye
            eye_roi_l = img[faces[0][159][1] - 10:faces[0][145][1] + 10, faces[0][33][0] - 10:faces[0][133][0] + 10]
            w_x = faces[0][133][0] - faces[0][33][0]
            w_y = faces[0][145][1] - faces[0][159][1]
            # img[0:20 + w_y, 0:20 + w_x] = eye_roi_l
            img[110:110 + 20 + w_y, 450:450 + 20 + w_x] = eye_roi_l

            cv2.rectangle(img, (450, 110), (470 + w_x, 130 + w_y),
                          colour, 2)

            # for right eye
            eye_roi_r = img[faces[0][386][1] - 10:faces[0][374][1] + 10, faces[0][362][0] - 10:faces[0][263][0] + 10]
            w_x = faces[0][263][0] - faces[0][362][0]
            w_y = faces[0][374][1] - faces[0][386][1]
            # print(w_y, w_x)
            img[110:110 + 20 + w_y, 760:760 + 20 + w_x] = eye_roi_r
            cv2.rectangle(img, (760, 110), (780 + w_x, 130 + w_y),
                          colour, 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 200), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)
        cv2.namedWindow("EBD", cv2.WINDOW_NORMAL)
        cv2.imshow("EBD", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()


if __name__ == "__main__":
    main()