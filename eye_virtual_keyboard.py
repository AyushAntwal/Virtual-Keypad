import cv2
import numpy as np
import Hand_Tracking_Module
from Hand_Tracking_Module import HandDetector
from time import sleep
from pynput.keyboard import Controller  #to provide output to external applications

from eye_blink_detector_module import FaceMeshDetector
from pygame import mixer
mixer.init()
voice_click = mixer.Sound('sclick.mp3')


cv2.namedWindow("Virtual_keyboard" , cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(0)

cap.set(3 , 1080)
cap.set(4 , 720)

detector = HandDetector(detectionCon=0.8)
eye_detector = FaceMeshDetector(maxFaces=1)

Keys = [["Q" , "W" , "E" , "R" , "T" , "Y" , "U" , "I" , "O" , "P"] ,
        ["A" , "S" , "D" , "F" , "G" , "H" , "J" , "K" , "L" , ";" ],
        ["Z" , "X" , "C" , "V" , "B" , "N" , "M" , "," , "." , "_" ]]

finalText = ""

# keyboard = Controller()      #to provide output to external applications

def drawALL(img , buttonList):

    for button in buttonList:  #this part is used to show keyboard buttons on screen
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, (x+100, y+100), (x + w+100, y + h+100), (0,0,0), cv2.FILLED)
        cv2.putText(img, button.text, (x + 15 + 100, y + 65+100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

    return img

class Button():
    def __init__(self , pos , text , size=[85,85]):
        self.pos = pos
        self.size = size
        self.text = text

buttonList = []
for i in range(len(Keys)):
    for j, Key in enumerate(Keys[i]):
        buttonList.append(Button([100 * j + 50, 100 * i + 50], Key))  #accessing position of keys


while True:
    success , img = cap.read()

    img1 = cv2.flip(img, 1)
    img = cv2.flip(img, 1)

    # print(img.shape)
    img, faces = eye_detector.findFaceMesh(img, False)
    if len(faces) != 0:
        blink, img1= eye_detector.EyeBlinkDetector(img, faces, True)
    img = detector.findHands(img)    #find hands in image

    ####################################################################################################
    # cv2.rectangle(img, (90, 110), (1200, 600), (182,121,60), cv2.FILLED)
    blur = cv2.GaussianBlur(img, (101,101), 0)
    mask = np.zeros((720,1280, 3), dtype=np.uint8)
    color = np.random.randint(low=255, high=256, size=3).tolist()
    img = np.where(mask == color, img, blur)
    #####################################################################################################

    img = drawALL(img, buttonList)   #calling buttons
    lmList = detector.findPosition(img) #notes landmarks
    if faces:
        colour = (0, 0, 200)
        if blink:
            colour = (0, 200, 0)
        cv2.putText(img, f'Blinked :{blink}', (535, 140), fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=2, color=colour, thickness=2)
        # for left eye
        eye_roi_l = img1[faces[0][159][1] - 10:faces[0][145][1] + 10, faces[0][33][0] - 10:faces[0][133][0] + 10]
        w_x = faces[0][133][0] - faces[0][33][0]
        w_y = faces[0][145][1] - faces[0][159][1]
        img[0:20 + w_y, 0:20 + w_x] = eye_roi_l
        #img[110:110 + 20 + w_y, 450:450+20 + w_x] = eye_roi_l

        cv2.rectangle(img, (450, 110), (470 + w_x, 130 + w_y),colour, 2)

        # for right eye
        eye_roi_r = img1[faces[0][386][1] - 10:faces[0][374][1] + 10, faces[0][362][0] - 10:faces[0][263][0] + 10]
        w_x = faces[0][263][0] - faces[0][362][0]
        w_y = faces[0][374][1] - faces[0][386][1]
        # print(w_y, w_x)
        img[110:110 + 20 + w_y, 760:760 + 20 + w_x] = eye_roi_r
        cv2.rectangle(img, (760, 110), (780 + w_x, 130 + w_y),colour, 2)

    #checking for hand:
    if lmList:
        for button in buttonList:
            x , y = button.pos
            w , h = button.size

            cv2.circle(img, (lmList[8][1], lmList[8][2]), 10, (0,0, 200), -1)
            if x+100 < lmList[8][1] < x+w+100 and y+100 < lmList[8][2] < y+h+100: #giving the x value of our index finger tip


                cv2.rectangle(img, (x+100, y+100), (x + w+100, y + h+100), (175 , 0 , 175), cv2.FILLED)
                cv2.putText(img, button.text, (x + 15 + 100, y + 65+100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)


                if blink:
                    voice_click.play()
                    cv2.rectangle(img, (x+100, y+100), (x + w+100, y + h+100), (0, 255, 0), cv2.FILLED)      # cv2.rectangle(img, button.text , (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 15 + 100, y + 65+100),                                          #(x + 15, y + 65)
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

                    finalText += button.text
                    # sleep(0.10)


    cv2.rectangle(img, (180,460), (1080,550), (0,0,0), cv2.FILLED)
    cv2.putText(img, finalText, (200,520),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
    sleep(0.11)   #working as senstivity controller while typing



    cv2.imshow("Virtual_keyboard" , img )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break