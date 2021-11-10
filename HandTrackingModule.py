import requests
import cv2
import numpy as np
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode = False, maxHands = 2, model_complexity = 1 ,minDetectCon = 0.5, minTrackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = model_complexity
        self.minDetectCon = minDetectCon
        self.minTrackCon = minTrackCon

        # Creating mpHands Object
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity,self.minDetectCon, self.minTrackCon)
        
        # To draw landmarks
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, img, draw = True):
        # Converting to RGB to work with mpHands
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        # Check if hand is detected or not
        if(self.results.multi_hand_landmarks):
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Draw the landmarks of hands
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNo = 0, draw = True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            # For loop for each landmark
            for id, lm in enumerate(myHand.landmark):
                # Check the coordinate of the landmarks
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                lmList.append([id, cx, cy])

                # Draw the landmark
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList            

def main():
    # Defining Variables
    url = "http://192.168.43.1:8080/shot.jpg"
    frame_width = 900
    frame_height = 560

    # Defining window
    cv2.namedWindow("Android Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Android Camera", frame_width, frame_height)

    # Retrieving the image from the url
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)

    # Check frame rate variable
    pTime = 0
    cTime = 0

    # Creating Hand Detector Object
    detector = handDetector()

    while True:
        # Retrieving the image from the url
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)

        # Detect hands and draw
        img = detector.findHands(img, draw=True)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        # Calculate the fps 
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)

        cv2.imshow("Android Camera", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()