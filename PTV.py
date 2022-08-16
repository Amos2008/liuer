import cv2
import matplotlib.pyplot as plt
import numpy as np
Vfile =  "F:/DJI.MOV"

def get_frames(filename):
    video = cv2.VideoCapture(filename)
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            return frame
        else:
            break
        video.release()
        return None

for f in get_frames(Vfile):
    if f is None:
        break
    cv2.imshow("frame", f)
    if cv2.waitKey(10) == 27:
        break
    cv2.destroyAllWindows()

def get_frame(filename, index):
    counter = 0
    video = cv2.VideoCapture(filename)
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            if counter == index:
                return frame
        else:
            break
        video.release()
        return None

frame = get_frame(Vfile, 80)
print("shape ", frame.shape)
print("pixel at (0,0), frame [0,0,:]")


def get_frames(filename):
    cap = cv2.VideoCapture(filename)
    while True:
        ret, frame = cap.read()
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(30)
        if key == 27:
            break
cap.release()
cv2.destroyAllWindows()