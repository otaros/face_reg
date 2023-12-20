import face_recognition
import cv2 as cv

video_capture = cv.VideoCapture(1)

while True:
    # show the video
    ret, frame = video_capture.read()
    small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
    small_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2GRAY)
    face_locations = face_recognition.face_locations(small_frame,model="cnn")
    # draw the rectangle
    print(face_locations)
    for top, right, bottom, left in face_locations:
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv.imshow("Video", frame)
    if (cv.waitKey(1) & 0xFF) == ord("q"):
        break
