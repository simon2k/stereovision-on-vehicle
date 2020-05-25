import serial
import glob

SERIAL_BAUD_RATE = 115200


class VehicleSerial(object):
    def __init__(self):
        self.serial = None

    def get_vehicle_serial(self):
        if not glob.glob('/dev/ttyAC*'):
            print('Waiting for the serial path to be present...')
            self.get_vehicle_serial()

        serial_path = glob.glob('/dev/ttyAC*')[0]

        try:
            self.serial = serial.Serial(serial_path, SERIAL_BAUD_RATE)
            self.serial.flush()
            self.wait_for_response('Ready')
        except serial.serialutil.SerialException as e:
            print('Serial exception: {}'.format(e))
            self.get_vehicle_serial()

    def wait_for_response(self, command):
        while True:
            if self.serial.in_waiting > 0:
                line = self.serial.readline()
                serialized_command = (command + '\r\n').encode()
                print('Line :', line, serialized_command)
                if serialized_command == line:
                    print('Confirmation: ', line)
                    break

    def send_command(self, command):
        serialized_command = command + '\n'
        self.serial.write(serialized_command.encode())
        self.wait_for_response(command)
