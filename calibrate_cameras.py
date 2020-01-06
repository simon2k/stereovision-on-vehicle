import numpy as np
from glob import glob
from computer_vision.camera_calibrator import CameraCalibrator

np.set_printoptions(precision=4, suppress=True)

CALIBRATION_STORAGE_PATH = 'data/calibration'
PHOTOS_FOLDER = 'data/photos/stereo/640x360'

right_images = glob(f'{PHOTOS_FOLDER}/right-*.jpg')

right_camera_calibrator = CameraCalibrator(right_images)
(right_reproj_err, right_camera_matrix, right_dist_coeffs) = right_camera_calibrator.calibrate()

print('Right camera')
print('Reprojection error: ', right_reproj_err)
print('Camera matrix: \n', right_camera_matrix)
print('Distortion coeff: \n', right_dist_coeffs)
print()

np.savez_compressed(f'{CALIBRATION_STORAGE_PATH}/right-camera-calibration.npz',
                    camera_matrix=right_camera_matrix,
                    dist_coeffs=right_dist_coeffs,
                    reproj_err=right_reproj_err)

left_images = glob(f'{PHOTOS_FOLDER}/left-*.jpg')
left_camera_calibrator = CameraCalibrator(left_images)
(left_reproj_err, left_camera_matrix, left_dist_coeffs) = left_camera_calibrator.calibrate()

print('Left camera')
print('Reprojection error: ', left_reproj_err)
print('Camera matrix: \n', left_camera_matrix)
print('Distortion coeff: \n', left_dist_coeffs)
print()

np.savez_compressed(f'{CALIBRATION_STORAGE_PATH}/left-camera-calibration.npz',
                    camera_matrix=left_camera_matrix,
                    dist_coeffs=left_dist_coeffs,
                    reproj_err=left_reproj_err)

# == Results on 06.01.2020 at 12:44:
#
# = Right camera
#
# Reprojection error:  0.098
#
# Camera matrix:
#  [[479.86   0.   305.22]
#  [  0.   479.71 173.99]
#  [  0.     0.     1.  ]]
#
# Distortion coeff:
# [[ 0.0415 -0.1536 -0.0052 -0.0051  0.0862  0.      0.      0.      0.002
#    0.0033  0.0059  0.0014]]

# 0.0415 & -0.1536 & -0.0052 & -0.0051 & 0.002 & 0.0033

# = Left camera
#
# Reprojection error:  0.1
#
# Camera matrix:
#  [[479.95   0.   339.72]
#  [  0.   479.57 176.  ]
#  [  0.     0.     1.  ]]
#
# Distortion coeff:
# [[ 0.0466 -0.1844 -0.012   0.0089  0.1147  0.      0.      0.     -0.0071
#    0.0003  0.0189  0.0023]]

# 0.0466 & -0.1844 & -0.012 & 0.0089 & -0.0071 & 0.0003
