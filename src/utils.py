from os.path import isdir
import numpy as np


def verify_folder_existence(path):
    if not isdir(path):
        raise Exception(f'Folder with this path: {path} does not exist')


# Source: https://www.geometrictools.com/Documentation/EulerAngles.pdf
def rotation_matrix_to_euler_angles(R):
    if R[0, 2] < 1:
        if R[0, 2] > -1:
            thetaY = np.arcsin(R[0, 2])
            thetaX = np.arctan2(-R[1, 2], R[2, 2])
            thetaZ = np.arctan2(-R[0, 1], R[0, 0])
        else:
            thetaY = -np.pi / 2
            thetaX = -np.arctan2(R[1, 0], R[1, 1])
            thetaZ = 0
    else:
        thetaY = np.pi / 2
        thetaX = np.arctan2(R[1, 0], R[1, 1])
        thetaZ = 0

    return np.degrees(np.array([thetaX, thetaY, thetaZ]))
