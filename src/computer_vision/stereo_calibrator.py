import cv2
import numpy as np
from computer_vision.checkerboard_points_finder import CheckerboardPointsFinder
from utils import rotation_matrix_to_euler_angles

np.set_printoptions(precision=4, suppress=True)


class StereoCalibrator:
    CALIBRATION_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-9)

    def __init__(self, right_image_paths, left_image_paths):
        self.right_image_paths = right_image_paths
        self.left_image_paths = left_image_paths

        left_camera_calibration = np.load('data/calibration/left-camera-calibration.npz')
        self.left_camera_matrix = left_camera_calibration['camera_matrix']
        self.left_dist_coef = left_camera_calibration['dist_coeffs']

        right_camera_calibration = np.load('data/calibration/right-camera-calibration.npz')
        self.right_camera_matrix = right_camera_calibration['camera_matrix']
        self.right_dist_coef = right_camera_calibration['dist_coeffs']

    def calibrate_stereo(self):
        (right_checkerboard_plane_corners, right_image_checkerboard_corners, _) = \
            CheckerboardPointsFinder.find_all_checkerboard_points(self.right_image_paths)
        (left_checkerboard_plane_corners, left_image_checkerboard_corners, img_size) = \
            CheckerboardPointsFinder.find_all_checkerboard_points(self.left_image_paths)

        print('\nMacierz wewnętrzna kamery lewej: \n', self.left_camera_matrix)
        print('\nParametry zniekształceń kamery lewej: \n', self.left_dist_coef)

        print('\nMacierz wewnętrzna kamery prawej: \n', self.right_camera_matrix)
        print('\nParametry zniekształceń kamery prawej: \n', self.right_dist_coef)

        (
            reproj_err,
            new_left_camera_matrix, new_left_dist_coeffs,
            new_right_camera_matrix, new_right_dist_coeffs,
            R, T, E, F,
            per_view_errors
        ) = \
            cv2.stereoCalibrateExtended(
                objectPoints=left_checkerboard_plane_corners,
                imagePoints1=left_image_checkerboard_corners,
                imagePoints2=right_image_checkerboard_corners,
                cameraMatrix1=self.left_camera_matrix,
                distCoeffs1=self.left_dist_coef,
                cameraMatrix2=self.right_camera_matrix,
                distCoeffs2=self.right_dist_coef,
                imageSize=img_size,
                R=None,
                T=None,
                E=None,
                F=None,
                flags=cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_THIN_PRISM_MODEL,
                criteria=self.CALIBRATION_CRITERIA
            )

        print('\nBłąd reprojekcji: ', reproj_err)

        print('\nNowa macierz wewnętrzna kamery lewej: \n', new_left_camera_matrix)
        print('\nNowe parametry zniekształceń kamery lewej: \n', new_left_dist_coeffs)

        print('\nNowa macierz wewnętrzna kamery prawej: \n', new_right_camera_matrix)
        print('\nNowe parametry zniekształceń kamery prawej: \n', new_right_dist_coeffs)

        print('\nMacierz rotacji:\n', R)
        print('\nStopnie X, Y, Z macierzy rotacji: \n', rotation_matrix_to_euler_angles(R))
        print('\nWektor translacji: \n', T)

        (
            left_rectified_rotation,
            right_rectified_rotation,
            left_rectified_projection,
            right_rectified_projection,
            Q,
            left_valid_roi,
            right_valid_roi
        ) = cv2.stereoRectify(
            cameraMatrix1=new_left_camera_matrix,
            distCoeffs1=new_left_dist_coeffs,
            cameraMatrix2=new_right_camera_matrix,
            distCoeffs2=new_right_dist_coeffs,
            imageSize=img_size[::-1],
            R=R,
            T=T,
            alpha=0,
            flags=cv2.CALIB_ZERO_DISPARITY
        )

        print('\nZrektyfikowana prawa macierz rotacji: \n', right_rectified_rotation)
        print('\nZrektyfikowana lewa macierz rotacji: \n', left_rectified_rotation)

        print('\nStopnie X, Y, Z zrektyfikowanej prawej macierzy rotacji: \n',
              rotation_matrix_to_euler_angles(left_rectified_rotation))

        print('\nStopnie X, Y, Z zrektyfikowanej lewej macierzy rotacji: \n',
              rotation_matrix_to_euler_angles(right_rectified_rotation))

        print('\nNowa macierz wewnętrzna kamery prawej: \n', right_rectified_projection)
        print('\nNowa macierz wewnętrzna kamery lewej: \n', left_rectified_projection)

        print('\nMacierz rzutowania punktów z określoną rozbieżnością na homogeniczny punkt 3W: \n', Q)

        (left_map1, left_map2) = cv2.initUndistortRectifyMap(
            cameraMatrix=new_left_camera_matrix,
            distCoeffs=new_left_dist_coeffs,
            R=left_rectified_rotation,
            newCameraMatrix=left_rectified_projection,
            size=img_size[::-1],
            m1type=cv2.CV_16SC2,
        )

        (right_map1, right_map2) = cv2.initUndistortRectifyMap(
            cameraMatrix=new_right_camera_matrix,
            distCoeffs=new_right_dist_coeffs,
            R=right_rectified_rotation,
            newCameraMatrix=right_rectified_projection,
            size=img_size[::-1],
            m1type=cv2.CV_16SC2,
        )

        return reproj_err, left_map1, left_map2, right_map1, right_map2, Q
