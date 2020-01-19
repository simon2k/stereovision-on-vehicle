import cv2

CAMERA_PATHS = {
    'left': '/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_ADBCD97F-video-index0',
    'right': '/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_84BD15AF-video-index0',
}


class Camera:
    def __init__(self, side, width, height):
        self.side = side
        self.camera = cv2.VideoCapture(CAMERA_PATHS[self.side], cv2.CAP_V4L)
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        has_frame, frame = self.camera.read()

        if not has_frame:
            raise Exception(f'Camera: {self.side} does not return any image')

        return frame

    def grab(self):
        self.camera.grab()

    def retrieve(self):
        _, frame = self.camera.retrieve()
        return frame

    def release(self):
        self.camera.release()
