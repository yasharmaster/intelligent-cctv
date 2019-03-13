import numpy as np
import cv2
import argparse
from yolo import Predictor
from helper import *

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', default='resources/dog.jpg',
                help = 'path to input image')
ap.add_argument('-c', '--config', default='resources/yolov3.cfg',
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', default='resources/yolov3.weights',
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', default='resources/yolov3.txt',
                help = 'path to text file containing class names')
args = ap.parse_args()

cap = cv2.VideoCapture('resources/dog.mp4')
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet(args.weights, args.config)

predictor = Predictor(classes, COLORS, net)

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_LINEAR)
    frame = predictor.predict(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    numpy_horizontal_concat = np.concatenate((gray, adjust_gamma(gray, 1.5)), axis=1)
    cv2.imshow('Main', numpy_horizontal_concat)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()