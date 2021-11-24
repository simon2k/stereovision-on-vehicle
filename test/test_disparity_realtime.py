import atexit
import cv2
import numpy as np
import computer_vision

np.set_printoptions(precision=5, suppress=True)

stereo_results_cache = {
    'disparity_mtx': None,
    'disparity_img': None,
    'depth_mtx': None,
}


def mouseCallback(event, x, y, *other):
    if event == cv2.EVENT_LBUTTONDOWN:
        disparity = str(stereo_results_cache['disparity_mtx'][y, x])
        threeD = str(stereo_results_cache['depth_mtx'][y, x])
        print(str((y, x)) + ' Rozbieżność: ' + disparity + ' 3W: ' + threeD)


cv2.namedWindow('disparity')
cv2.setMouseCallback('disparity', mouseCallback)

rtDispGen = computer_vision.RealtimeDisparityGenerator()

while True:
    disparity_mtx, depth_mtx, disparity_img = rtDispGen.generate_depth_mtxs()

    stereo_results_cache['disparity_mtx'] = disparity_mtx
    stereo_results_cache['disparity_img'] = disparity_img
    stereo_results_cache['depth_mtx'] = depth_mtx

    cv2.imshow('disparity', disparity_img)

    key = cv2.waitKey(5)

    if key == 27:
        break


@atexit.register
def finish_program():
    rtDispGen.finish()
    cv2.destroyAllWindows()
