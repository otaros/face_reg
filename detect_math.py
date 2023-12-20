import cv2 as cv
import time


cap = cv.VideoCapture(0, cv.CAP_DSHOW)
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_alt.xml")

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    font = cv.FONT_HERSHEY_SIMPLEX
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    new_frame_time = time.time()
    detected = face_cascade.detectMultiScale(grayscale, 1.1, 9)
    for x, y, w, h in detected:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    # converting the fps into integer
    fps = int(fps)
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
    # putting the FPS count on the frame
    cv.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv.LINE_AA)
    cv.imshow("Video", frame)
    if (cv.waitKey(1) & 0xFF) == ord("q"):
        break
