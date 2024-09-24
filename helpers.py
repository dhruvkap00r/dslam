import numpy as np

def add_ones(x):
 if len(x.shape) == 1:
   return np.concatenate([x,np.array([1.0])], axis=0)
 else:
   return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def kp_norm(pts, K):
  return np.dot(np.linalg.inv(K), add_ones(pts).T).T[:,0:2]

def kp_denorm(pts, K):
  ret = np.dot(K, np.array([pts[0], pts[1], 1.0]))
  return int(round(ret[0])), int(round(ret[1]))
