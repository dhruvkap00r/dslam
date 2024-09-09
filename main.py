#!/usr/bin/python3.10
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import itertools
from skimage.transform import EssentialMatrixTransform
import cv2
from Extractor import Extract
import numpy as np
import matplotlib.pyplot as plt
"""
1. detect features
2. map features on a graph
3. Basically create a 3D space like tesla has.

"""
import time
cap = cv2.VideoCapture('test_countryroad.mp4')





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

def Mapping(frame, kp, des, prev_kp, prev_des):
  #connect every landmarks
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = bf.match(des,prev_des)
  matches = sorted([m for m in matches if m.distance < 10], key=lambda m: m.distance)
  #get landmark locations (coords)
  x1, y1 = [], [] 
  x2, y2 = [], []
  ret = []
  try:
    for m in matches:
      kp1 = kp[m.queryIdx].pt
      kp2 = kp[m.trainIdx].pt
      
      ret.append((kp1, kp2))
  except:
    pass
  prev_kp = kp
  prev_des = des
  if len(ret) > 0:
    ret = np.array(ret)
    ret[:, 0, :] = kp_norm(ret[:,0,:])
    ret[:, 1, :] = kp_norm(ret[:,1,:])
    if len(ret[:,0]) > 8 and len(ret[:,1]) > 8:
      try:
        model, inliers = ransac((ret[:, 0], ret[:, 1]),
                              EssentialMatrixTransform,
                              #FundamenalMatrixTransform,
                              min_samples=8,
                              residual_threshold=0.05,
                              max_trials=100)
        if np.sum(inliers) == None:
          print("no")
        else:
          T = Extract(frame).extractRt(model.params)


        ret = ret[inliers]
      except Exception as error:
        print(error)
  return ret

  #make assumptions

def display(frame, kp, ret):
  for p in kp:
    x, y = map(lambda x: int(round(x)), p.pt)
    cv2.circle(frame,(x,y), 2, (0,255,0), 1)
  for point in ret:
    x1, y1 = kp_denorm(point[0])
    x2, y2 =  kp_denorm(point[1])
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.line(frame,(x1, y1),(x2,y2), (0,0,255), 1)

#from Frame import Display
    


#frame loop

prev_kp = [] 
prev_des = []
while True:
  ret, frame = cap.read()
  if not ret:
    break
  kp,des = Extract(frame).extract_features()
  matches = []
  if len(prev_kp) > 0:

    matches = (Mapping(frame, kp,des,prev_kp, prev_des))
  else:
    prev_kp = kp
    prev_des = des
  
  display(frame, kp, matches)
  cv2.imshow('frame', frame)
  if cv2.waitKey(1) == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()
