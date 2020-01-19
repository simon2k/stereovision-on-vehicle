import cv2
from os import path
from computer_vision.camera import Camera
from computer_vision.checkerboard_points_finder import CheckerboardPointsFinder
from utils import verify_folder_existence


class PhotosCapture:
    PHOTOS_PATH = '../data/photos/separate'

    def __init__(self, side, width, height):
        self.side = side
        self.width = width
        self.height = height

        folder = path.dirname(path.realpath(__file__))
        self.photos_folder = path.join(folder, self.PHOTOS_PATH)
        verify_folder_existence(self.photos_folder)

    def start_capturing(self):
        camera = Camera(self.side, self.width, self.height)

        images_count = 0

        while True:
            frame = camera.read()

            cv2.imshow(f'{self.side}-photo', frame)

            key = cv2.waitKey(5)

            if key == 27:
                break

            if key == 32:
                has_corners, corners = CheckerboardPointsFinder.find_points(frame)

                if not has_corners:
                    print('Exception: The image has no corners')
                else:
                    print(f'captured image {images_count}')

                    cv2.imwrite(f'{self.photos_folder}/{self.side}-{images_count}.jpg', frame)
                    images_count += 1

        cv2.destroyAllWindows()
