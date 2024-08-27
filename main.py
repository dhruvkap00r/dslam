from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import itertools
from skimage.transform import EssentialMatrixTransform
import cv2
import numpy as np
import matplotlib.pyplot as plt
"""
1. detect features
2. map features on a graph
3. Basically create a 3D space like tesla has.

"""
import time
cap = cv2.VideoCapture('test_ohio.mp4')
orb = cv2.ORB_create(nfeatures=100, scaleFactor=1.2)





if not cap.isOpened():
  exit()
sift = cv2.SIFT_create()

HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

Cx = HEIGHT//2
Cy = WIDTH//2
f_x = 0
f_y = 0
camera_pose = [[f_x,0, Cx],#setting skew=0
               [0, f_y, Cy],
               [0, 0, 1]]






def extract_features(frame):
  kp = cv2.goodFeaturesToTrack(np.mean(frame, axis=2).astype(np.uint8), 3000, 0.001, 3)
  #kp, des = sift.detectAndCompute(frame, None)
  kp = np.intp(kp)
  kps = [cv2.KeyPoint(float(c[0][0]), float(c[0][1]), 1) for c in kp]
  kps, des = orb.compute(frame, kps)
  return kps, des

def pose(x,y):
  return camera_pose*x, camera_pase*y


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
      #x1, y1 = kp1
      #x2, y2 = kp2
      #x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
      ret.append((kp1, kp2))
  except:
    pass 
  prev_kp = kp
  prev_des = des
  if len(ret) > 0:
    ret = np.array(ret)
    if len(ret[:,0]) > 8 and len(ret[:,1]) > 8:
      try:
        model, inliers = ransac((ret[:, 0], ret[:, 1]),
                              EssentialMatrixTransform,
                              min_samples=8,
                              residual_threshold=2,
                              max_trials=10)
        ret = ret[inliers]
      except:
        pass
  return ret

  #make assumptions

def pose_estimation(ret):
  #first we need an initial state to start with

  


  
def display(frame, kp, ret):
  for p in kp:
    x, y = map(lambda x: int(round(x)), p.pt)
    cv2.circle(frame,(x,y), 2, (0,255,0), 1)
  try:
    for point in ret:
      x1, y1 = point[0]
      x2, y2 = point[1]
      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
      cv2.line(frame,(x1, y1),(x2,y2), (0,0,255), 1)
  except:
    print("no match")

     
    


#frame loop

prev_kp = [] 
prev_des = []
while True:
  ret, frame = cap.read()
  if not ret:
    break
  kp,des = extract_features(frame)
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
