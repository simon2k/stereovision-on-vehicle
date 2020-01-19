import atexit
import cv2
import numpy as np
import time
from computer_vision.camera import Camera

np.set_printoptions(precision=5, suppress=True)

stereo_calibration = np.load('data/calibration/stereo-calibration.npz')
left_map1 = stereo_calibration['left_map1']
left_map2 = stereo_calibration['left_map2']
right_map1 = stereo_calibration['right_map1']
right_map2 = stereo_calibration['right_map2']
Q = stereo_calibration['Q']

stereo_result = {
    'disparity': None,
    'img_disparity': None,
    'img_depth': None,
}

stereo_config = {
    'stereo_method': 0,
    'num_disparities': 8,
    'block_size': 9,
    'min_disparity': 0,
    'uniqueness_ratio': 0,
    'speckle_range': 0,
    'speckle_window_size': 2,
    'disp_12_max_diff': 0,
    'bm_pre_filter_type': 0,
    'sgbm_mode': 0,
    'color': 0
}

images = {
    'left_rectified_img': None,
    'right_rectified_img': None,
    'times': [],
    'count': 0,
}

KRAW_POJ_LEWA = -40
KRAW_POJ_PRAWA = 160


def rectify_left(img):
    return cv2.remap(src=img, map1=left_map1, map2=left_map2, interpolation=cv2.INTER_LINEAR)


def rectify_right(img):
    return cv2.remap(src=img, map1=right_map1, map2=right_map2, interpolation=cv2.INTER_LINEAR)


def create_stereo_processor(config):
    num_disparities = (config['num_disparities'] + 1) * 16
    block_size = config['block_size'] * 2 - 1 + 6
    stereo_processor = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)

    stereo_processor.setPreFilterType(config['bm_pre_filter_type'])
    stereo_processor.setMinDisparity(config['min_disparity'])
    stereo_processor.setUniquenessRatio(config['uniqueness_ratio'])
    stereo_processor.setSpeckleRange(config['speckle_range'])
    stereo_processor.setSpeckleWindowSize(config['speckle_window_size'])
    stereo_processor.setDisp12MaxDiff(config['disp_12_max_diff'])

    return stereo_processor


stereo_processor = create_stereo_processor(stereo_config)


def make_decision():
    disparity = stereo_processor.compute(images['left_rectified_img'], images['right_rectified_img'])
    disparity = disparity / 16
    disparity = disparity.astype('int16')

    stereo_result['disparity'] = disparity

    img_disparity = np.copy(disparity)
    img_depth = cv2.reprojectImageTo3D(img_disparity, Q)

    stereo_result['img_depth'] = img_depth

    (min_val, max_val, _, _) = cv2.minMaxLoc(img_disparity)
    if max_val and (max_val - min_val):
        img_disparity = cv2.convertScaleAbs(img_disparity, alpha=(255 / (max_val - min_val)))

    points_before_left_end = stereo_result['img_depth'][:, :, 0] > KRAW_POJ_LEWA
    points_before_right_end = stereo_result['img_depth'][:, :, 0] < KRAW_POJ_PRAWA
    points_in_window_track_width = points_before_left_end & points_before_right_end

    points_below_ceiling = stereo_result['img_depth'][:, :, 1] > -10
    points_above_floor = stereo_result['img_depth'][:, :, 1] < 90
    points_before_fov_end = stereo_result['img_depth'][:, :, 2] < 700
    points_before_infinity = stereo_result['img_depth'][:, :, 2] > 0
    points_in_full_fov = points_below_ceiling & points_above_floor & points_before_fov_end & points_before_infinity

    points_in_main_track = points_in_window_track_width & points_in_full_fov

    min_obstacle_points = 1200
    has_obstacle = np.count_nonzero(points_in_main_track) > min_obstacle_points

    num_disparities = (stereo_config['num_disparities'] + 1) * 16

    track_disparity = np.copy(img_disparity)
    track_disparity[points_in_main_track == False] = 0
    track_disparity = track_disparity[:, num_disparities:]

    full_width_disparity = np.copy(img_disparity)
    full_width_disparity[points_in_full_fov == False] = 0
    full_width_disparity = full_width_disparity[:, num_disparities:]

    upper_images = np.concatenate(
        (images['left_rectified_img'][:, num_disparities:], img_disparity[:, num_disparities:]),
        axis=1
    )
    bottom_images = np.concatenate(
        (full_width_disparity, track_disparity),
        axis=1
    )
    all_images = np.concatenate((upper_images, bottom_images), axis=0)

    cv2.imshow('all', all_images)
    images['count'] += 1

    if has_obstacle:
        direction = None
        iteration = None

        full_fov_width = 560
        step = 20
        iterations = int((full_fov_width / 2) / step)

        for i in range(iterations):
            iteration = i + 1

            left_window_start = iteration * (-step) + KRAW_POJ_LEWA
            left_window_end = iteration * (-step) + KRAW_POJ_PRAWA

            left_window_points = \
                (stereo_result['img_depth'][:, :, 0] > left_window_start) & \
                (stereo_result['img_depth'][:, :, 0] < left_window_end) & \
                points_in_full_fov

            right_window_start = iteration * step + KRAW_POJ_LEWA
            right_window_end = iteration * step + KRAW_POJ_PRAWA

            right_window_points = \
                (stereo_result['img_depth'][:, :, 0] > right_window_start) & \
                (stereo_result['img_depth'][:, :, 0] < right_window_end) & \
                points_in_full_fov

            has_left_area_free = np.count_nonzero(left_window_points) < min_obstacle_points
            has_right_area_free = np.count_nonzero(right_window_points) < min_obstacle_points

            if has_right_area_free:
                direction = 'right'
                break

            if has_left_area_free:
                direction = 'left'
                break

        if direction:
            window_shift = iteration * step

            avg_distance = np.mean(stereo_result['img_depth'][points_in_main_track][:, 2])
            tg_theta = window_shift / avg_distance

            angle = np.degrees(tg_theta)

            print('direction: {} angle: {}'.format(direction, angle))
        else:
            vehicle_middle = (abs(KRAW_POJ_LEWA) + KRAW_POJ_PRAWA) / 2 + KRAW_POJ_LEWA

            left_half_window_points_mask = (stereo_result['img_depth'][:, :, 0] < vehicle_middle) & points_in_full_fov
            right_half_window_points_mask = (stereo_result['img_depth'][:, :, 0] > vehicle_middle) & points_in_full_fov

            avg_left_distances_sum = np.mean(stereo_result['img_depth'][left_half_window_points_mask][:, 2])
            avg_right_distances_sum = np.mean(stereo_result['img_depth'][right_half_window_points_mask][:, 2])

            next_direction = 'left' if avg_left_distances_sum > avg_right_distances_sum else 'right'
            angle = 30

            print('cannot avoid the obstacle - turns {} {} degrees)'.format(next_direction, angle))
    else:
        print('No obstacle - continuing forward!')


left_camera = Camera('left', 640, 360)
right_camera = Camera('right', 640, 360)


def capture_and_make_decision():
    start = time.time()

    right_camera.grab()
    left_camera.grab()

    print('Time for capturing photos: {}'.format(time.time() - start))

    right_frame = right_camera.retrieve()
    left_frame = left_camera.retrieve()

    if right_frame is None:
        print('Right frame is None')
        capture_and_make_decision()

    if left_frame is None:
        print('Left frame is None')
        capture_and_make_decision()

    right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
    left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)

    images['left_rectified_img'] = rectify_left(left_frame)
    images['right_rectified_img'] = rectify_right(right_frame)

    make_decision()
    images['times'].append(time.time() - start)


while True:
    capture_and_make_decision()

    key = cv2.waitKey(5)

    if key in [27, 32]:
        print('Avg time for movement: {}'.format(np.mean(images['times'])))
        break


@atexit.register
def finish_program():
    right_camera.release()
    left_camera.release()
    cv2.destroyAllWindows()
