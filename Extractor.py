
#!/usr/bin/python3.10
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import itertools
from skimage.transform import EssentialMatrixTransform
import cv2
import numpy as np
import matplotlib.pyplot as plt


orb = cv2.ORB_create(nfeatures=100, scaleFactor=1.2)
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
    
  def extractRt(self, m):
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
