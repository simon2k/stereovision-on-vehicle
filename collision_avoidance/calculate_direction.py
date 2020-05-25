import cv2
import numpy as np

MIN_OBSTACLE_POINTS = 3000

KRAW_POJ_PRAWA = 170
KRAW_POJ_LEWA = -60


def calculate_direction(depth_mtx, disparity_img):
    points_after_left_end = depth_mtx[:, :, 0] > KRAW_POJ_LEWA
    points_before_right_end = depth_mtx[:, :, 0] < KRAW_POJ_PRAWA
    points_in_window_track_width = points_after_left_end & points_before_right_end

    points_below_ceiling = depth_mtx[:, :, 1] > -10
    points_above_floor = depth_mtx[:, :, 1] < 130
    points_before_fov_end = depth_mtx[:, :, 2] < 600
    points_before_infinity = depth_mtx[:, :, 2] > 10
    points_in_full_fov = points_below_ceiling & points_above_floor & points_before_fov_end & points_before_infinity

    points_in_main_track = points_in_window_track_width & points_in_full_fov

    print('# Points: ', np.count_nonzero(points_in_main_track))

    has_obstacle = np.count_nonzero(points_in_main_track) > MIN_OBSTACLE_POINTS

    # track_disparity = np.copy(disparity_img)
    # track_disparity[points_in_main_track == False] = 0

    # full_width_disparity = np.copy(disparity_img)
    # full_width_disparity[points_in_full_fov == False] = 0

    # images = np.concatenate((track_disparity, full_width_disparity), axis=1)
    # cv2.imshow('all', images)

    if has_obstacle:
        direction = None
        iteration = None

        full_fov_width = 400  # FOV width in centimeters
        step = 20
        iterations = int((full_fov_width / 2) / step)

        for i in range(iterations):
            iteration = i + 1

            left_window_start = iteration * (-step) + KRAW_POJ_LEWA
            left_window_end = iteration * (-step) + KRAW_POJ_PRAWA

            left_window_points = \
                (depth_mtx[:, :, 0] > left_window_start) & \
                (depth_mtx[:, :, 0] < left_window_end) & \
                points_in_full_fov

            right_window_start = iteration * step + KRAW_POJ_LEWA
            right_window_end = iteration * step + KRAW_POJ_PRAWA

            right_window_points = \
                (depth_mtx[:, :, 0] > right_window_start) & \
                (depth_mtx[:, :, 0] < right_window_end) & \
                points_in_full_fov

            has_left_area_free = np.count_nonzero(left_window_points) < MIN_OBSTACLE_POINTS
            has_right_area_free = np.count_nonzero(right_window_points) < MIN_OBSTACLE_POINTS

            if has_right_area_free:
                direction = 'right'
                break

            if has_left_area_free:
                direction = 'left'
                break

        if direction:
            window_shift = iteration * step

            avg_distance = np.mean(depth_mtx[points_in_main_track][:, 2])
            print('Avg distance: ', avg_distance)
            tg_theta = window_shift / avg_distance

            angle = np.degrees(tg_theta)
        else:
            vehicle_middle = (abs(KRAW_POJ_LEWA) + KRAW_POJ_PRAWA) / 2 + KRAW_POJ_LEWA

            left_half_window_points_mask = (depth_mtx[:, :, 0] < vehicle_middle) & points_in_full_fov
            right_half_window_points_mask = (depth_mtx[:, :, 0] > vehicle_middle) & points_in_full_fov

            avg_left_distances_sum = np.mean(depth_mtx[left_half_window_points_mask][:, 2])
            avg_right_distances_sum = np.mean(depth_mtx[right_half_window_points_mask][:, 2])

            direction = 'left' if avg_left_distances_sum > avg_right_distances_sum else 'right'
            angle = 30

        return {'direction': direction, 'angle': angle}
    else:
        return {'direction': 'forward', 'angle': 0}
