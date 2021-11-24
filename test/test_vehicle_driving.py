import vehicle

vehicle_serial = vehicle.VehicleSerial()
vehicle_serial.get_vehicle_serial()

vehicle = vehicle.Vehicle(vehicle_serial)

vehicle.turn_left(12)

vehicle.move_forward(5)

vehicle.turn_right(12)
