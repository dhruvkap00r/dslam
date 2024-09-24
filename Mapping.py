
#!/usr/bin/python3.10
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import itertools
from skimage.transform import EssentialMatrixTransform
import cv2
from Extractor import Extract
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
def Mapping(frame, kp, des, prev_kp, prev_des, K):
  #connect every landmarks
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = bf.match(des,prev_des)
  matches = sorted([m for m in matches if m.distance < 10], key=lambda m: m.distance)
  #get landmark locations (coords)
  x1, y1 = [], [] 
  x2, y2 = [], []
  T = []
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
    ret[:, 0, :] = kp_norm(ret[:,0,:], K)
    ret[:, 1, :] = kp_norm(ret[:,1,:], K)
    if ret[:,0].shape[0] > 8:
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
        pass
  return ret, T




#Triangulation

def Triangulation(kp, T, K):
  if len(T) < 4:
    return
    
  T = np.array(T)
  T = T[:3, :4]
  T = np.dot(K, T)
  t = cv2.triangulatePoints(T, T,kp[0], kp[1])
  ret = np.zeros((2,3))
  kp = t[:3] / t[3]
  ret[0] = kp[:3, 0]
  ret[1] = kp[:3, 1]
  return ret

