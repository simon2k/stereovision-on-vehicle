import numpy as np
from glob import glob
from computer_vision.camera_calibrator import CameraCalibrator

np.set_printoptions(precision=4, suppress=True)

CALIBRATION_STORAGE_PATH = 'data/calibration'
PHOTOS_FOLDER = 'data/photos/separate'

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

# Right camera
# Reprojection error:  0.093
# Camera matrix:
#  [[478.0344   0.     311.4095]
#  [  0.     477.7713 172.3884]
#  [  0.       0.       1.    ]]
# Distortion coeff:
#  [[ 0.0469 -0.1802 -0.0013  0.0004  0.1359  0.      0.      0.     -0.0024
#   -0.0014 -0.0037  0.0039]]
#
# 478.03 & 477.77 & 311.41 & 172.39
# 0.0469 & -0.1802 & -0.0013 & 0.0004 & -0.0024 & -0.0014
#
# Left camera
# Reprojection error:  0.083
# Camera matrix:
#  [[477.9134   0.     333.2866]
#  [  0.     477.4695 177.9162]
#  [  0.       0.       1.    ]]
# Distortion coeff:
#  [[ 0.0392 -0.1556  0.0006  0.0082  0.1037  0.      0.      0.     -0.0131
#   -0.001  -0.0043  0.0015]]

# 477.91 & 477.47 & 333.29 & 177.92
# 0.0392 & -0.1556 & 0.0006 & 0.0082 & -0.0131 & -0.001
