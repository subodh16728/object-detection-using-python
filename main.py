#importing modules
import cv2 
import numpy as np

#setting global parametres
thres = 0.45 # Threshold to detect object if more than thres
nms_threshold = 0.2 #detects overlapping of objects 
cap = cv2.VideoCapture(0) #capture video frame
cap.set(3,1280) #window size
# cap.set(4,720)
# cap.set(10,150)

#generate all classNames
classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#Declare model path
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

#Declare model 
net = cv2.dnn_DetectionModel(weightsPath,configPath) #Load model as per configuration and weight
net.setInputSize(320,320) #resize input image
net.setInputScale(1.0/ 127.5) #Normalize the data
net.setInputMean((127.5, 127.5, 127.5)) #Set Input mean
net.setInputSwapRB(True) #RGB to BGR as opencv works on BGR

#confs is probability of bounding box belonging to a particular class
while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres) #Detecting with model
    bbox = list(bbox) 
    confs = list(np.array(confs).reshape(1,-1)[0]) #Resize confidence array
    confs = list(map(float,confs)) #converting to float
    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold) #Selecting bounding box indices as per threshold
    # print(classNames)
    for i in indices:
        # print(i)
        # i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        # print(classIds[i])
        # print(classNames[classIds[i]-1])
        cv2.putText(img,classNames[classIds[i]-1].upper() + " " +  str(round(confs[i] * 100,1)),(x-10,y),
        cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2) #Add class text to output

        cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2) #Creae bbox in output

    cv2.imshow('Output',img) #Update the output frame
    print("Running")
    cv2.waitKey(2) #Checks after 2ms 