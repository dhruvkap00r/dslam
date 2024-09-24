#!/usr/bin/python3.10
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import itertools
from skimage.transform import EssentialMatrixTransform
import cv2
from Extractor import Extract
import numpy as np
import matplotlib.pyplot as plt
from Frame import Display
from Mapping import *
from helpers import *
from multiprocessing import Process, Queue

"""
1. detect features
2. map features on a graph
3. Basically create a 3D space like tesla has.

"""
import time
cap = cv2.VideoCapture('test_ohio.mp4')





if not cap.isOpened():
  exit()
sift = cv2.SIFT_create()

HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

Cx = HEIGHT//2
Cy = WIDTH//2
F =  512 

#K is intrinsic matrix

K = np.array([[F,0, Cx],#setting skew=0
                        [0, F, Cy],
                        [0, 0, 1]])




def display(frame, kp, ret):
  for p in kp:
    x, y = map(lambda x: int(round(x)), p.pt)
    cv2.circle(frame,(x,y), 2, (0,255,0), 1)
  for point in ret:
    x1, y1 = kp_denorm(point[0], K)
    x2, y2 =  kp_denorm(point[1], K)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.line(frame,(x1, y1),(x2,y2), (0,0,255), 1)





#frame loop

dis = Display()
prev_kp = [] 
prev_des = []
while True:
  ret, frame = cap.read()
  if not ret:
    break
  kp,des = Extract(frame).extract_features()
  matches = []
  pose = []
  if len(prev_kp) > 0:

    matches, pose = Mapping(frame, kp,des,prev_kp, prev_des, K)
  else:
    prev_kp = kp
    prev_des = des
  
  new_coords = Triangulation(matches, pose, K)
 
  if new_coords is not None: 
    dis.addtoqueue(pose,new_coords)
  display(frame, kp, matches)



  cv2.imshow('frame', frame)
  if cv2.waitKey(1) == ord('q'):
    q.put(None)
    break
  


cap.release()
cv2.destroyAllWindows()
