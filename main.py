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


camera_pose = np.array([[F,0, Cx],#setting skew=0
                        [0, F, Cy],
                        [0, 0, 1]])


def add_ones(x):
 if len(x.shape) == 1:
   return np.concatenate([x,np.array([1.0])], axis=0)
 else:
   return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def kp_norm(pts):
  return np.dot(np.linalg.inv(camera_pose), add_ones(pts).T).T[:,0:2]

def kp_denorm(pts):
  ret = np.dot(camera_pose, np.array([pts[0], pts[1], 1.0]))
  return int(round(ret[0])), int(round(ret[1]))


def display(frame, kp, ret):
  for p in kp:
    x, y = map(lambda x: int(round(x)), p.pt)
    cv2.circle(frame,(x,y), 2, (0,255,0), 1)
  for point in ret:
    x1, y1 = kp_denorm(point[0])
    x2, y2 =  kp_denorm(point[1])
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.line(frame,(x1, y1),(x2,y2), (0,0,255), 1)


#frame loop

prev_kp = [] 
prev_des = []
while True:
  ret, frame = cap.read()
  if not ret:
    break
  kp,des = Extract(frame).extract_features()
  global matches
  matches = []
  global pose
  if len(prev_kp) > 0:

    matches, pose = (Mapping(frame, kp,des,prev_kp, prev_des))
    new_coords = Triangulation(matches, pose)
  else:
    prev_kp = kp
    prev_des = des
  
  
  display(frame, kp, matches)
  cv2.imshow('frame', frame)
  if cv2.waitKey(1) == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()
