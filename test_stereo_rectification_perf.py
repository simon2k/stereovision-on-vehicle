import cv2
import numpy as np
import time

stereo_calibration = np.load('data/calibration/stereo-calibration.npz')
left_map1 = stereo_calibration['left_map1']
left_map2 = stereo_calibration['left_map2']
right_map1 = stereo_calibration['right_map1']
right_map2 = stereo_calibration['right_map2']


def rectify_left(img):
    return cv2.remap(src=img, map1=left_map1, map2=left_map2, interpolation=cv2.INTER_LINEAR)


def rectify_right(img):
    return cv2.remap(src=img, map1=right_map1, map2=right_map2, interpolation=cv2.INTER_LINEAR)


left_img = cv2.imread('data/photos/stereo/left-4.jpg')
right_img = cv2.imread('data/photos/stereo/right-4.jpg')

start = time.time()

rect_left_img = rectify_left(left_img)
rect_right_img = rectify_right(right_img)

diff = time.time() - start

print('Czas dla naprawy pary obrazów [ms]', np.round(diff * 1e3, 3))
print('Napraw w ciągu jednej sekundy:', int(1 / diff))
