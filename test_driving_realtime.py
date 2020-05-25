import atexit
import cv2
import numpy as np
import computer_vision
import collision_avoidance
import vehicle

np.set_printoptions(precision=5, suppress=True)

stereo_results_cache = {
    'disparity_mtx': None,
    'disparity_img': None,
    'depth_mtx': None,
}


def mouseCallback(event, x, y, *other):
    if event == cv2.EVENT_LBUTTONDOWN:
        disparity = str(stereo_results_cache['disparity_mtx'][y, x])
        threeD = str(stereo_results_cache['depth_mtx'][y, x])
        print(str((y, x)) + ' Rozbieżność: ' + disparity + ' 3W: ' + threeD)


cv2.namedWindow('disparity')
cv2.setMouseCallback('disparity', mouseCallback)

rtDispGen = computer_vision.RealtimeDisparityGenerator()

vehicle_serial = vehicle.VehicleSerial()
vehicle_serial.get_vehicle_serial()

vehicle = vehicle.Vehicle(vehicle_serial)

MAX_STEPS = 4
steps = 0

while steps < MAX_STEPS:
    disparity_mtx, depth_mtx, disparity_img = rtDispGen.generate_depth_mtxs()

    stereo_results_cache['disparity_mtx'] = disparity_mtx
    stereo_results_cache['disparity_img'] = disparity_img
    stereo_results_cache['depth_mtx'] = depth_mtx

    cv2.imshow('disparity', disparity_img)

    direction = collision_avoidance.calculate_direction(depth_mtx, disparity_img)

    print('Direction: ', direction)

    vehicle.move(direction)
    steps += 1

    key = cv2.waitKey(5)

    if key == 27:
        break


@atexit.register
def finish_program():
    rtDispGen.finish()
    cv2.destroyAllWindows()
