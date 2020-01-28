import atexit
import cv2
import numpy as np
import computer_vision

np.set_printoptions(precision=5, suppress=True)

stereo_calibration = np.load('data/calibration/stereo-calibration.npz')
left_map1 = stereo_calibration['left_map1']
left_map2 = stereo_calibration['left_map2']
right_map1 = stereo_calibration['right_map1']
right_map2 = stereo_calibration['right_map2']
Q = stereo_calibration['Q']

stereo_results_cache = {
    'disparity_mtx': None,
    'disparity_img': None,
    'depth_mtx': None,
}


def rectify_left(img):
    return cv2.remap(src=img, map1=left_map1, map2=left_map2, interpolation=cv2.INTER_LINEAR)


def rectify_right(img):
    return cv2.remap(src=img, map1=right_map1, map2=right_map2, interpolation=cv2.INTER_LINEAR)


def mouseCallback(event, x, y, *other):
    if event == cv2.EVENT_LBUTTONDOWN:
        disparity = str(stereo_results_cache['disparity_mtx'][y, x])
        threeD = str(stereo_results_cache['depth_mtx'][y, x])
        print(str((y, x)) + ' Rozbieżność: ' + disparity + ' Dystans: ' + threeD)


disparity_calculator = computer_vision.DisparityCalculator(Q)


def capture(left_camera, right_camera):
    right_camera.grab()
    left_camera.grab()

    right_frame = right_camera.retrieve()
    left_frame = left_camera.retrieve()

    if right_frame is None:
        print('Right frame is None')
        capture()

    if left_frame is None:
        print('Left frame is None')
        capture()

    right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
    left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)

    blended = cv2.addWeighted(left_frame, 0.5, right_frame, 0.5, 0.0)

    disparity_mtx, depth_mtx, disparity_img = disparity_calculator.calculate(left_frame, right_frame)

    stereo_results_cache['disparity_mtx'] = disparity_mtx
    stereo_results_cache['disparity_img'] = disparity_img
    stereo_results_cache['depth_mtx'] = depth_mtx

    cv2.imshow('blended', blended)
    cv2.imshow('disparity', disparity_img)


left_camera = computer_vision.Camera('left', 640, 360)
right_camera = computer_vision.Camera('right', 640, 360)

cv2.namedWindow('blended')
cv2.namedWindow('disparity')
cv2.setMouseCallback('disparity', mouseCallback)

while True:
    capture(left_camera, right_camera)

    key = cv2.waitKey(5)

    if key in [27, 32]:
        break


@atexit.register
def finish_program():
    right_camera.release()
    left_camera.release()
    cv2.destroyAllWindows()
