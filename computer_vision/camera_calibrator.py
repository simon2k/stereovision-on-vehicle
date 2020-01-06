import cv2
import numpy as np
from computer_vision.checkerboard_points_finder import CheckerboardPointsFinder, \
    X_CORNERS, Y_CORNERS, \
    CELL_WIDTH, CELL_HEIGHT, \
    BOARD_WIDTH, BOARD_HEIGHT



class CameraCalibrator:
    CALIBRATION_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, int(1e3), 1e-6)
    CALIBRATION_FLAGS = cv2.CALIB_USE_LU | cv2.CALIB_THIN_PRISM_MODEL

    def __init__(self, image_paths):
        self.image_paths = image_paths

    def find_all_checkerboard_points(self):
        obj_p_zero = np.zeros((X_CORNERS * Y_CORNERS, 3), np.float32)
        board_grid = np.mgrid[0:BOARD_WIDTH:CELL_WIDTH, 0:BOARD_HEIGHT:CELL_HEIGHT]
        obj_p_zero[:, :2] = board_grid.T.reshape(-1, 2)
        checkerboard_plane_corners = []
        image_checkerboard_corners = []
        image_size = None

        for file_name in self.image_paths:
            img = cv2.imread(file_name)
            image_size = img.shape[:-1]
            has_corners, corners = CheckerboardPointsFinder.find_points(img)

            if has_corners:
                checkerboard_plane_corners.append(obj_p_zero)
                image_checkerboard_corners.append(corners)
            else:
                print(f'The file {file_name} is excluded since it has no corners')

        return checkerboard_plane_corners, image_checkerboard_corners, image_size

    def calibrate(self):
        (checkerboard_plane_corners, image_checkerboard_corners, img_size) = self.find_all_checkerboard_points()

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
