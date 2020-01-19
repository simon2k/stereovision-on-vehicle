import cv2
import numpy as np

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

rect_left_img = rectify_left(left_img)
rect_right_img = rectify_right(right_img)

blended = cv2.addWeighted(right_img, 0.5, left_img, 0.5, 0.0)
rect_blended = cv2.addWeighted(rect_right_img, 0.5, rect_left_img, 0.5, 0.0)

# cv2.imshow('rect-left', rect_left_img)
# cv2.imshow('rect-right', rect_right_img)

cv2.imshow('before-rect-left', left_img)
cv2.imshow('before-rect-right', right_img)

cv2.imshow('blended-before-rect', blended)
cv2.imshow('blended-after-rect', rect_blended)

while True:
    key = cv2.waitKey(5)
    if key == 27:
        break

cv2.destroyAllWindows()
