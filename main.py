#!/usr/bin/python3.10
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
cap = cv2.VideoCapture('sample.mp4')
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
camera_pose = np.array([[f_x,0, Cx],#setting skew=0
                        [0, f_y, Cy],
                        [0, 0, 1]])
class Extract:
  def __init__(self, frame):
    self.frame = frame


  def extract_features(self):
    kp = cv2.goodFeaturesToTrack(np.mean(self.frame, axis=2).astype(np.uint8), 3000, 0.001, 3)
    #kp, des = sift.detectAndCompute(frame, None)
    kp = np.intp(kp)
    kps = [cv2.KeyPoint(float(c[0][0]), float(c[0][1]), 1) for c in kp]
    kps, des = orb.compute(self.frame, kps)
    return kps, des

  def pose(x,y):
    return camera_pose*x, camera_pase*y
    
  def extractRt(m):
    W = np.asmatrix([[0,-1,0],
                [1,0,0],
                [0,0,1]])
    
    Z = np.asmatrix([[0,1,0],
                [-1,0,0],
                [0,0,1]])

    U,E,VT = np.linalg.svd(m)
    R = np.dot(np.dot(U, np.linalg.inv(W)), VT)
    t = U[:,2]
    T = np.eye(4)
    T[:3,3] = t
    T[:3, :3] = R
    return T



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
  T = np.zeros((4,4))
  prev_kp = kp
  prev_des = des
  if len(ret) > 0:
    ret = np.array(ret)
    if len(ret[:,0]) > 8 and len(ret[:,1]) > 8:
      try:
        model, inliers = ransac((ret[:, 0], ret[:, 1]),
                              EssentialMatrixTransform,
                              #FundamenalMatrixTransform,
                              min_samples=8,
                              residual_threshold=2,
                              max_trials=100)
        if np.sum(inliers) == None:
          print("no")
        else:
          T = extract(model.params)


        ret = ret[inliers]
      except Exception as error:
        pass
  return ret

  #make assumptions


  


  
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
  #  dis = Display()
   # dis.frame(matches)
  else:
    prev_kp = kp
    prev_des = des
  print(frame[-1].shape)
  
  display(frame, kp, matches)
  cv2.imshow('frame', frame)
  if cv2.waitKey(1) == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()
