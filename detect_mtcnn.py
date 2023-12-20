from mtcnn import MTCNN
import cv2 as cv
import time

video_capture = cv.VideoCapture(0, cv.CAP_DSHOW)
video_capture.set(cv.CAP_PROP_FPS, 30)
detector = MTCNN(scale_factor=0.5)

# used to record the time when we processed last frame 
prev_frame_time = 0
  
# used to record the time at which we processed current frame 
new_frame_time = 0

max_fps = 0
min_fps = 1000
avg_fps = []

while video_capture.isOpened():
    ret, frame = video_capture.read()
    font = cv.FONT_HERSHEY_SIMPLEX 
    small_frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
    small_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)
    new_frame_time = time.time() 
    result = detector.detect_faces(small_frame)
    for face in result:
        x1, y1, width, height = face["box"]
        x1 *= 2
        y1 *= 2
        width *= 2
        height *= 2
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
        # converting the fps into integer 

    avg_fps.append(fps)
    
    fps = int(fps) 

    if fps > max_fps:
        max_fps = fps
    if fps < min_fps and fps > 0:
        min_fps = fps
  
    # converting the fps to string so that we can display it on frame 
    # by using putText function 
    fps = str(fps) 
  
    # putting the FPS count on the frame 
    cv.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv.LINE_AA) 
    cv.imshow("Video", frame)
    if (cv.waitKey(1) & 0xFF) == ord("q"):
        print("Max FPS: ", max_fps)
        print("Min FPS: ", min_fps)
        print("Avg FPS: ", sum(avg_fps)/len(avg_fps))
        break