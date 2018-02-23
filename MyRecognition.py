import cv2
import os
import numpy as np
import sys
import math
from matplotlib import pyplot as plt

def ReadCalibration(path):
    #pixels = []
    #meters = []
    #angles = []
    data = []
    file = open(path, 'r')
    for line in file:
        pixel, meter = (
            item.strip() for item in line.split(' ', 2))
        set = []
        #pixels.append(pixel)
        #meters.append(meter)
        #angles.append(angle)
        set.append(float(pixel))
        set.append(float(meter))
        #set.append(float(angle))
        data.append(set)
    print(data)
    return data

def FaceRec(frame):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    myPath= os.path.dirname(os.path.realpath(__file__))+'\\'+'opencv_frame_0.png'
    #img = cv2.imread(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imwrite(myPath, frame)

def MonoSLAM(frame):
    kalman=cv2.KalmanFilter()

def DetectSpot(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    labels = cv2.connectedComponents(thresh)[1]
    mask = np.zeros(thresh.shape, dtype="uint8")
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    x, y = maxLoc;
    #print(x,y)
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if labelMask[y, x] == 255 and numPixels<1500:
            mask = cv2.add(mask, labelMask)
    hsv = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)
    lower_red = np.array([173,90,120])
    upper_red = np.array([180,255,255])
    #frame = mask
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([0,90,120])
    upper_red = np.array([7,255,255])
    mask3 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.add(mask2, mask3)
    mask = cv2.add(mask, mask2)
    gauss = cv2.GaussianBlur(mask, (19, 19), 0)
    thresh = cv2.threshold(gauss, 127, 255, cv2.THRESH_BINARY)[1]
    mask = thresh
    im2, cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    return cnts, mask

def testInterpolate(x1, y1, x2, y2, x):
    y = y1 + ((y2 - y1)/(x2 - x1))*(x - x1)
    return y

def CalculateDistance(pix, mylist):
    for data in mylist:
        if pix == data[0]:
            return data[1]
    x1Val = 0
    y1Val = sys.maxsize
    x2Val = sys.maxsize
    y2Val = 0
    for data in mylist:
        if (data[0] > pix) and (data[0] < x2Val):
            x2Val = data[0]
            y2Val = data[1]
    for data in mylist:
        if (data[0] < pix) and (data[0] > x1Val):
            x1Val = data[0]
            y1Val = data[1]
    if (x1Val == 0) and (y1Val == sys.float_info):
        x1Val = x2Val
        y1Val = y2Val
        x2Val = sys.maxsize
        for data in mylist:
            if (data[0] > x1Val) and (data[0] < x2Val):
                x2Val = data[0]
                y2Val = data[1]
    if (x2Val == sys.maxsize) and (y2Val == 0):
        x2Val = x1Val
        y2Val = y1Val
        x1Val = 0
        for data in mylist:
            if (data[0] < x2Val) and (data[0] > x1Val):
                x1Val = data[0]
                y1Val = data[1]
    return testInterpolate(x1Val, y1Val, x2Val, y2Val, pix)

def CalculatePixels(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def GetPoints(mask, minx, maxx, miny, maxy):
    TwoDList = []
    for i in range(miny, maxy):
        minj = maxx
        maxj = minx
        finalx = -1
        minxfound = False
        maxxfound = False
        for j in range(minx, maxx):
            pixel = mask.item(i, j)
            if pixel == 255:
                if  j< minj:
                    minj = j
                    minxfound = True
                if j > maxj:
                    maxj = j
                    maxxfound = True
        pair = []
        if minxfound and maxxfound:
            finalx = minj+(maxj-minj)/2
        if finalx > 0:
            pair.append(finalx)
            pair.append(i)
            TwoDList.append(pair)
    return TwoDList


def GetCoords(mylist, frame):
    distances = []
    height, width, _ = frame.shape
    for item in mylist:
        distances.append(CalculatePixels(width/2, item[1], item[0], item[1]))
    return distances


def GetDistances(mylist, data):
    distances = []
    for item in mylist:
        dist = CalculateDistance(item, data)*math.cos(0.523599)
        distances.append(dist)
    return distances
