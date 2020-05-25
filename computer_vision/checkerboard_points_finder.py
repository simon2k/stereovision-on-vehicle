import cv2
import numpy as np

CELL_WIDTH = 29.3
CELL_HEIGHT = 29.3
X_CORNERS = 7
Y_CORNERS = 5
BOARD_WIDTH = CELL_WIDTH * X_CORNERS
BOARD_HEIGHT = CELL_HEIGHT * Y_CORNERS
BOARD_SIZE = (X_CORNERS, Y_CORNERS)
CORNERS_FINDER_FLAGS = cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE


class CheckerboardPointsFinder:
    @staticmethod
    def find_points(image):
        return cv2.findChessboardCornersSB(image, BOARD_SIZE, flags=CORNERS_FINDER_FLAGS)

    @staticmethod
    def find_all_checkerboard_points(image_paths):
        obj_p_zero = np.zeros((X_CORNERS * Y_CORNERS, 3), np.float32)
        board_grid = np.mgrid[0:BOARD_WIDTH:CELL_WIDTH, 0:BOARD_HEIGHT:CELL_HEIGHT]
        obj_p_zero[:, :2] = board_grid.T.reshape(-1, 2)
        checkerboard_plane_corners = []
        image_checkerboard_corners = []
        image_size = None

        for file_name in image_paths:
            img = cv2.imread(file_name)
            image_size = img.shape[:-1]
            has_corners, corners = CheckerboardPointsFinder.find_points(img)

            if has_corners:
                checkerboard_plane_corners.append(obj_p_zero)
                image_checkerboard_corners.append(corners)
            else:
                print(f'The file {file_name} is excluded since it has no corners')

        return checkerboard_plane_corners, image_checkerboard_corners, image_size
