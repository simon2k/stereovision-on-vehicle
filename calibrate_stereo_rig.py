from glob import glob
from computer_vision.stereo_calibrator import StereoCalibrator
import numpy as np

np.set_printoptions(precision=4, suppress=True)

PHOTOS_FOLDER = 'data/photos/stereo'

right_image_paths = glob(f'{PHOTOS_FOLDER}/right-*.jpg')
right_image_paths.sort()
left_image_paths = glob(f'{PHOTOS_FOLDER}/left-*.jpg')
left_image_paths.sort()

stereo_calibrator = StereoCalibrator(right_image_paths=right_image_paths, left_image_paths=left_image_paths)

(reproj_err, left_map1, left_map2, right_map1, right_map2, Q) = stereo_calibrator.calibrate_stereo()

np.savez_compressed(
    'data/calibration/stereo-calibration.npz',
    reproj_err=reproj_err,
    left_map1=left_map1,
    left_map2=left_map2,
    right_map1=right_map1,
    right_map2=right_map2,
    Q=Q
)

# Dane startowe:
#
#
# Macierz wewnętrzna kamery lewej:
#  [[477.9134   0.     333.2866]
#  [  0.     477.4695 177.9162]
#  [  0.       0.       1.    ]]
#
# Parametry zniekształceń kamery lewej:
#  [[ 0.0392 -0.1556  0.0006  0.0082  0.1037  0.      0.      0.     -0.0131
#   -0.001  -0.0043  0.0015]]
#
# Macierz wewnętrzna kamery prawej:
#  [[478.0344   0.     311.4095]
#  [  0.     477.7713 172.3884]
#  [  0.       0.       1.    ]]
#
# Parametry zniekształceń kamery prawej:
#  [[ 0.0469 -0.1802 -0.0013  0.0004  0.1359  0.      0.      0.     -0.0024
#   -0.0014 -0.0037  0.0039]]
#
#
# Rezultat kalibracji kamer:
#
# Błąd reprojekcji:  0.2187230303137086
#
# Nowa macierz wewnętrzna kamery lewej:
#  [[477.9134   0.     333.2866]
#  [  0.     477.4695 177.9162]
#  [  0.       0.       1.    ]]
#
# Nowe parametry zniekształceń kamery lewej:
#  [[ 0.0392 -0.1556  0.0006  0.0082  0.1037  0.      0.      0.     -0.0131
#   -0.001  -0.0043  0.0015  0.      0.    ]]
#
# Nowa macierz wewnętrzna kamery prawej:
#  [[478.0344   0.     311.4095]
#  [  0.     477.7713 172.3884]
#  [  0.       0.       1.    ]]
#
# Nowe parametry zniekształceń kamery prawej:
#  [[ 0.0469 -0.1802 -0.0013  0.0004  0.1359  0.      0.      0.     -0.0024
#   -0.0014 -0.0037  0.0039  0.      0.    ]]
#
# Macierz rotacji:
#  [[ 0.9999 -0.0122  0.0082]
#  [ 0.0121  0.9999  0.0096]
#  [-0.0083 -0.0095  0.9999]]
#
# Stopnie X, Y, Z macierzy rotacji:
#  [-0.5458  0.4755  0.6923]
#
# Wektor translacji:
#  [[-95.564 ]
#  [  0.2494]
#  [ -0.3915]]
#
#
#
#
#
# Zrektyfikowana prawa macierz rotacji:
#  [[ 1.     -0.0026  0.0041]
#  [ 0.0026  1.     -0.0048]
#  [-0.0041  0.0048  1.    ]]
#
# Zrektyfikowana lewa macierz rotacji:
#  [[ 0.9998 -0.0148  0.0123]
#  [ 0.0148  0.9999  0.0049]
#  [-0.0123 -0.0047  0.9999]]
#
# Stopnie X, Y, Z zrektyfikowanej prawej macierzy rotacji:
#  [-0.2687^\circ \\ 0.7062^\circ \\ 0.8453^\circ]
#
# Stopnie X, Y, Z zrektyfikowanej lewej macierzy rotacji:
#  [0.2742^\circ \\ 0.234^\circ \\  0.1506^\circ]
#
#
#
# Nowa macierz wewnętrzna kamery prawej:
#  [[   492.0936      0.        317.5933 -47026.9846]
#  [     0.        492.0936    174.2858      0.    ]
#  [     0.          0.          1.          0.    ]]
#
#
# Nowa macierz wewnętrzna kamery lewej:
#  [[492.0936   0.     317.5933   0.    ]
#  [  0.     492.0936 174.2858   0.    ]
#  [  0.       0.       1.       0.    ]]
#
#
#
# Macierz rzutowania punktów z określoną rozbieżnością na homogeniczny punkt 3W:
#  [[   1.        0.        0.     -317.5933]
#  [   0.        1.        0.     -174.2858]
#  [   0.        0.        0.      492.0936]
#  [   0.        0.        0.0105   -0.    ]]
