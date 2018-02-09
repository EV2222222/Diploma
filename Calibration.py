import numpy as np
import cv2
import glob


cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#images = glob.glob('grid.bmp')

while True:
    ret, frame = cam.read()
    #cv2.imshow("test", frame)
    if not ret:
        break
    img = np.copy(frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret2, corners = cv2.findChessboardCorners(gray, (9,6))

    # If found, add object points, image points (after refining them)
    k = cv2.waitKey(1)
    if ret2 == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners2,ret)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    cv2.imshow("test", img)
    if k % 256 == 32:
        cv2.imwrite("original.jpg",frame)
        cv2.imwrite("result.jpg",img)
        h,  w = img.shape[:2]
        ret3, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1])
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite('undistorted.png',dst)

cv2.destroyAllWindows()
