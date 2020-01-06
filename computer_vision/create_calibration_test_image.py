import cv2
import numpy as np

LINE_COLOR = (0, 0, 0)
LINE_WIDTH = 1
HEIGHT_STEP = 9
WIDTH_STEP = 16

blank_image = np.zeros((360, 640, 3), np.uint8)
blank_image[:, :] = (255, 255, 255)

points = np.array([np.arange(16, 625, WIDTH_STEP), np.arange(9, 352, HEIGHT_STEP)]).T

start_points = points[:int(points.shape[0] / 2), :]
end_points = np.flip(points[int(points.shape[0] / 2):, :])


def draw_line(start_point, end_point):
    cv2.line(blank_image, start_point, end_point, LINE_COLOR, LINE_WIDTH)


for i in np.arange(start_points.shape[0]):
    start_point = start_points[i]
    end_point = end_points[i]

    draw_line((start_point[0], start_point[1]), (start_point[0], end_point[0]))
    draw_line((end_point[1], start_point[1]), (end_point[1], end_point[0]))
    draw_line((start_point[0], start_point[1]), (end_point[1], start_point[1]))
    draw_line((start_point[0], end_point[0]), (end_point[1], end_point[0]))

cv2.imwrite('data/photos/calibration_test.png', blank_image)
