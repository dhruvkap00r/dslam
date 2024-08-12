
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
print
if not cap.isOpened():
  exit()
sift = cv2.SIFT_create()

HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))




def extract_features(frame):
  kp = cv2.goodFeaturesToTrack(np.mean(frame, axis=2).astype(np.uint8), 3000, 0.001, 3)
  #kp, des = sift.detectAndCompute(frame, None)
  kp = np.intp(kp)
  kps = [cv2.KeyPoint(float(c[0][0]), float(c[0][1]), 1) for c in kp]
  kps, des = orb.compute(frame, kps)
  return kps, des

def Mapping(frame, kp, des, prev_kp, prev_des):
  #connect every landmarks
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = bf.match(des,prev_des)
  print("f")
  matches = sorted([m for m in matches if m.distance < 10], key=lambda m: m.distance)
  #get landmark locations (coords)
  x1, y1 = [], [] 
  x2, y2 = [], []
  try:
    for m in matches:

      x1, y1 = kp[m.queryIdx].pt
      x2, y2 = kp[m.trainIdx].pt
      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
      cv2.line(frame, (x1, y1), (x2, y2), (0,0,255), 1)
  except:
    pass 
  prev_kp = kp
  prev_des = des
 
  
  return matches

  #make assumptions
   
  #observe
  #we need match features from previous image to current one.
  #update

  
def display(frame, kp):

  for p in kp:
    x,y = map(lambda x: int(round(x)), p.pt)
    cv2.circle(frame,(x,y), 2, (0,255,0), 1)
    



#frame loop

prev_kp = [] 
prev_des = []
while True:
  ret, frame = cap.read()
  if not ret:
    break
  kp,des = extract_features(frame)
  if len(prev_kp) > 0:

    matches = Mapping(frame, kp,des,prev_kp, prev_des)

  else:
    prev_kp = kp
    prev_des = des
  
  display(frame, kp)
  cv2.imshow('frame', frame)
  if cv2.waitKey(1) == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()
