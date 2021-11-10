import requests
import cv2
import numpy as np
import mediapipe as mp
import time
import HandTrackingModule as htm

# Defining Variables
url = "http://192.168.43.1:8080/shot.jpg"
frame_width = 900
frame_height = 560

window_name = "Android Camera"
board_name = "Drawing Board"

# Defining window
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, frame_width, frame_height)

# Retrieving the image from the url
img_resp = requests.get(url)
img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
img = cv2.imdecode(img_arr, -1)

pTime = 0

# Drawing Board
cv2.namedWindow(board_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(board_name, frame_width, frame_height)
board_height = 1080
board_width = 1920
green_board = np.zeros((board_height, board_width, 3), np.uint8)

detector = htm.handDetector(minDetectCon=0.8)

while True:
    # Retrieving the image from the url
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)

    img = detector.findHands(img, draw = False)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList) != 0:
        _, index_cx, index_cy = lmList[8]
        cv2.circle(img, (index_cx, index_cy), 15, (0, 255, 0), cv2.FILLED)
        cv2.circle(green_board, (index_cx, index_cy), 15, (0, 255, 255), cv2.FILLED)

    # Frame Per Second
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)

    cv2.imshow(board_name, green_board)
    cv2.imshow(window_name, img)
    cv2.waitKey(1)