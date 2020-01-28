import cv2


class DisparityCalculator(object):
    def __init__(self, Q):
        stereo_processor = cv2.StereoBM_create(numDisparities=80, blockSize=11)
        stereo_processor.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
        stereo_processor.setMinDisparity(90)
        self.stereo_processor = stereo_processor
        self.Q = Q

    def calculate(self, left_img, right_img):
        disparity_mtx = self.stereo_processor.compute(left_img, right_img)
        disparity_mtx = disparity_mtx / 16
        disparity_mtx = disparity_mtx.astype('int16')

        depth_mtx = cv2.reprojectImageTo3D(disparity_mtx, self.Q, handleMissingValues=0)
        disparity_img = 255 - cv2.convertScaleAbs(disparity_mtx - 90)

        return disparity_mtx, depth_mtx, disparity_img
