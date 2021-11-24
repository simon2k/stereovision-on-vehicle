import cv2
import numpy as np
import computer_vision


class RealtimeDisparityGenerator(object):
    def __init__(self):

        stereo_calibration = np.load('data/calibration/stereo-calibration.npz')
        self.left_map1 = stereo_calibration['left_map1']
        self.left_map2 = stereo_calibration['left_map2']
        self.right_map1 = stereo_calibration['right_map1']
        self.right_map2 = stereo_calibration['right_map2']
        Q = stereo_calibration['Q']

        self.disparity_calculator = computer_vision.DisparityCalculator(Q)

        self.left_camera = computer_vision.Camera('left', 640, 360)
        self.right_camera = computer_vision.Camera('right', 640, 360)

    def rectify_left(self, img):
        return cv2.remap(src=img, map1=self.left_map1, map2=self.left_map2, interpolation=cv2.INTER_LINEAR)

    def rectify_right(self, img):
        return cv2.remap(src=img, map1=self.right_map1, map2=self.right_map2, interpolation=cv2.INTER_LINEAR)

    def generate_depth_mtxs(self):
        self.right_camera.grab()
        self.left_camera.grab()

        right_frame = self.right_camera.retrieve()
        left_frame = self.left_camera.retrieve()

        if right_frame is None:
            print('Right frame is None')
            self.generate_depth_mtxs()

        if left_frame is None:
            print('Left frame is None')
            self.generate_depth_mtxs()

        right_frame = self.rectify_right(cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY))
        left_frame = self.rectify_left(cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY))

        return self.disparity_calculator.calculate(left_frame, right_frame)

    def finish(self):
        self.right_camera.release()
        self.left_camera.release()
