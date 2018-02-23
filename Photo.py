import numpy as np
import cv2
import platform
from MyRecognition import *
import sys
import plyfile as ply

def takePicture():
    list = ReadCalibration("test.txt")
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        contours, mask = DetectSpot(frame)
        #framecopy = frame.copy()
        frame, centers, mylist = DrawContours(contours, frame, mask)
        cv2.imshow("test", frame)
        if not ret:
            break

        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            points = GetCoords(mylist, frame)
            print(points)
            distances = GetDistances(points, list)
            print(distances)
            img_counter += 1
    cam.release()
    cv2.destroyAllWindows()

def DrawContours(cnts, frame, mask):
    cX = 0
    cY = 0
    centers = []
    height, width, _ = frame.shape
    min_x, min_y = width, height
    max_x = max_y = 0
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        #((cX, cY), radius) = cv2.minEnclosingCircle(c)
        #cv2.drawContours(frame, c, -1, (0, 0, 255), 3)
    if max_x - min_x > 0 and max_y - min_y > 0:
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
    # = cv2.boundingRect((min_x, min_y), (max_x, max_y))
    list = GetPoints(mask, min_x, max_x, min_y, max_y)
    for item in list:
        cv2.circle(frame,(int(item[0]), int(item[1])), 1, (0, 0, 255),-1)
    return frame, centers, list

def saveToFile():
    ply.PlyElement.header

takePicture()