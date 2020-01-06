import cv2
from os import path
from computer_vision.camera import Camera
from computer_vision.checkerboard_points_finder import CheckerboardPointsFinder
from utils import verify_folder_existence


class StereoPhotosCapture:
    PHOTOS_PATH = '../data/photos/stereo'

    def __init__(self, width, height):
        self.width = width
        self.height = height

        folder = path.dirname(path.realpath(__file__))
        self.photos_folder = path.join(folder, f'{self.PHOTOS_PATH}/{width}x{height}')
        verify_folder_existence(self.photos_folder)

    def start_capturing(self):
        right_camera = Camera('right', self.width, self.height)
        left_camera = Camera('left', self.width, self.height)

        images_count = 0

        while True:
            right_frame = right_camera.read()
            left_frame = left_camera.read()

            cv2.imshow('right', right_frame)
            cv2.imshow('left', left_frame)

            key = cv2.waitKey(5)

            if key == 27:
                break

            if key == 32:
                has_right_corners, right_corners = CheckerboardPointsFinder.find_points(right_frame)
                has_left_corners, left_corners = CheckerboardPointsFinder.find_points(left_frame)

                if not has_right_corners:
                    print('Exception: The right image has no corners')
                elif not has_left_corners:
                    print('Exception: The left image has no corners')
                else:
                    print(f'captured image {images_count}')

                    cv2.imwrite(f'{self.photos_folder}/right-{images_count}.jpg', right_frame)
                    cv2.imwrite(f'{self.photos_folder}/left-{images_count}.jpg', left_frame)
                    images_count += 1

        cv2.destroyAllWindows()
