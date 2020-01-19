import cv2
from computer_vision.checkerboard_points_finder import CheckerboardPointsFinder


class CameraCalibrator:
    CALIBRATION_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, int(1e3), 1e-6)
    CALIBRATION_FLAGS = cv2.CALIB_THIN_PRISM_MODEL

    def __init__(self, image_paths):
        self.image_paths = image_paths

    def calibrate(self):
        (checkerboard_plane_corners, image_checkerboard_corners, img_size) = \
            CheckerboardPointsFinder.find_all_checkerboard_points(self.image_paths)

        reprojectionError, \
        cameraMatrix, distCoeffs, \
        rvecs, tvecs, \
        newObjPoints, \
        stdDeviationsIntrinsics, stdDeviationsExtrinsics, \
        stdDeviationsObjPoints, perViewErrors = \
            cv2.calibrateCameraROExtended(
                objectPoints=checkerboard_plane_corners,
                imagePoints=image_checkerboard_corners,
                imageSize=img_size,
                iFixedPoint=1,
                cameraMatrix=None,
                distCoeffs=None,
                criteria=self.CALIBRATION_CRITERIA,
                flags=self.CALIBRATION_FLAGS
            )

        return reprojectionError, cameraMatrix, distCoeffs
