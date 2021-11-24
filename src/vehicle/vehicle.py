import time

TIME_PER_10_CM_FORWARD = 0.44
TIME_PER_90_DEG_LEFT = 0.52
TIME_PER_90_DEG_RIGHT = 0.52
SPEED = 200


class Vehicle(object):
    def __init__(self, vehicle_serial):
        self.vehicle_serial = vehicle_serial

    def move_forward(self, distance):
        movement_time = distance / 10 * TIME_PER_10_CM_FORWARD
        self.vehicle_serial.send_command('f{speed}'.format(speed=SPEED))
        time.sleep(movement_time)
        self.stop()

    def stop(self):
        self.vehicle_serial.send_command('s')

    def turn_left(self, degrees):
        movement_time = degrees / 90 * TIME_PER_90_DEG_LEFT
        self.vehicle_serial.send_command('l{speed}'.format(speed=SPEED))
        time.sleep(movement_time)
        self.stop()

    def turn_right(self, degrees):
        movement_time = degrees / 90 * TIME_PER_90_DEG_RIGHT
        self.vehicle_serial.send_command('r{speed}'.format(speed=SPEED))
        time.sleep(movement_time)
        self.stop()

    def move(self, direction):
        if direction['direction'] == 'forward':
            self.move_forward(10)
        if direction['direction'] == 'right':
            self.turn_right(direction['angle'])
        if direction['direction'] == 'left':
            self.turn_left(direction['angle'])
