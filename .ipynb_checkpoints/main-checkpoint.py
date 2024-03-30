from tracker import *
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import pickle

# model = YOLO('yolov8n.pt')

# with open('cached_model.pkl', 'wb') as f:
#     pickle.dump(model, f)

with open('cached_model.pkl','rb') as f:
    model = pickle.load(f)

# print(model)



                
class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
              'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


tracker = Tracker()
count = 0

list =[]

cap=cv2.VideoCapture('highway_mini.mp4')

while True:
    ret,frame = cap.read()
    if not ret:
        print('breaking no ret')
        break
    count +=1
    frame = cv2.resize(frame,(1020,500))
    results = model.predict(frame)
    a = results[0].boxes.data
    a = a.detach().cpu().numpy()
    px = pd.DataFrame(a).astype("float")
    # print(px)
    for index,row in px.iterrows():
        # print(row)
        x1 = int(row[0])
        y1 =int(row[0])
        x2 = int(row[0])
        y2 = int(row[0])





    cv2.imshow('test',frame)
    if cv2.waitKey(0)&0xFF=='q':
        break

cap.release()
cv2.destroyAllWindows()
