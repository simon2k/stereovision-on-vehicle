from computer_vision.photos_capture import PhotosCapture

photo_capture = PhotosCapture(side='right', width=640, height=360)
photo_capture.start_capturing()

photo_capture = PhotosCapture(side='left', width=640, height=360)
photo_capture.start_capturing()
