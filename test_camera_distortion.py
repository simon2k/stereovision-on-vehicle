import cv2
import numpy as np

PHOTOS_PATH = 'data/photos/'

right_calibration = np.load('data/calibration/right-camera-calibration.npz')
right_camera_matrix = right_calibration['camera_matrix']
right_dist_coeffs = right_calibration['dist_coeffs']

right_img = cv2.imread(f'{PHOTOS_PATH}/calibration_test.png')
right_green_img = cv2.imread(f'{PHOTOS_PATH}/green_calibration_test.png')

right_img_corr = cv2.undistort(right_img, cameraMatrix=right_camera_matrix, distCoeffs=right_dist_coeffs)
blended = cv2.addWeighted(right_green_img, 0.5, right_img_corr, 0.5, 0.0)

cv2.imshow('right-original', right_img)
cv2.imshow('right-corrected', right_img_corr)
cv2.imshow('right-blended', blended)

left_calibration = np.load('data/calibration/left-camera-calibration.npz')
left_camera_matrix = left_calibration['camera_matrix']
left_dist_coeffs = left_calibration['dist_coeffs']

left_img = cv2.imread(f'{PHOTOS_PATH}/calibration_test.png')
left_img_corr = cv2.undistort(left_img, cameraMatrix=left_camera_matrix, distCoeffs=left_dist_coeffs)
blended = cv2.addWeighted(left_img, 0.8, left_img_corr, 0.8, 0.0)

cv2.imshow('left-original', left_img)
cv2.imshow('left-corrected', left_img_corr)
cv2.imshow('left-blended', left_img_corr)

while True:
    key = cv2.waitKey(5)
    if key == 27:
        break

cv2.destroyAllWindows()
