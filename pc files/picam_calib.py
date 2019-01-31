
"""
OpenCV-Python Tutorials - Camera Calibration and 3D Reconstruction
http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
"""

import cv2
import numpy as np
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 6x8 chess board, prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

# 3d point in real world space
objpoints = []
# 2d points in image plane
imgpoints = []
h, w = 0, 0

images = glob.glob('picam/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # find chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

    # add object points, image points
    if ret:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # draw and display the corners
        cv2.drawChessboardCorners(img, (8, 6), corners, ret)
        cv2.imshow('image', img)
        cv2.waitKey(500)

# calibration
retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

print "camera matrix:\n", cameraMatrix

# pi camera intrinsic parameters
ay = cameraMatrix[1, 1]
u0 = cameraMatrix[0, 2]
v0 = cameraMatrix[1, 2]
print "Ay:", ay
print "u0:", u0
print "v0:", v0

cv2.destroyAllWindows()