#!/usr/bin/env python3.10
import pypangolin as pango
from OpenGL.GL import *
import numpy as np
from multiprocessing import Process, Queue




class Display:
  def __init__(self):
    self.Height = 640
    self.Width = 480
    self.q = Queue()
    self.process = Process(target=self.startp)
    self.process.start()

  def startp(self):
    self.main_thread()
    while 1:
      self.viewer_refresh()

  def main_thread(self):

    self.win = pango.CreateWindowAndBind("projection", self.Height, self.Width)
    glEnable(GL_DEPTH_TEST)

    #for both projectionmatrix and modelviewat i need insentric matrix
    #size, focal-point, principle axis, Znear, Zfar

    pm = pango.ProjectionMatrix(self.Width, self.Height, 
                                    420, 420, 
                                    self.Width//2, self.Height//2,
                                    0.1, 1000)
    #camera position and look-at points
    #T = None
    mv = pango.ModelViewLookAt(-0, 0.5, -3, 0, 0, 0, pango.AxisY)
    self.s_cam = pango.OpenGlRenderState(pm, mv)

    handler = pango.Handler3D(self.s_cam)
    self.d_cam = (pango.CreateDisplay()
              .SetBounds(
                      pango.Attach(0),
                      pango.Attach(1),
                      pango.Attach.Pix(180),
                      pango.Attach(1),
                      -(self.Height / self.Width)
               )
               .SetHandler(handler)
             )


  def viewer_refresh(self):
    
    kps,pose = self.q.get()
    """
    if kps.all() == None:
      return
    if isinstance(kps, np.ndarray):
      if kps.ndim > 0:
        kps = [np.array([[kp[0]], [kp[1]],[kp[2]]], dtype=np.float64) for kp in kps]

    else:
      if len(kps) != None:

        kps = kps
      else: 
        return
      """
    OpenGL.GL.glClear(OpenGL.GL.GL_COLOR_BUFFER_BIT | OpenGL.GL.GL_DEPTH_BUFFER_BIT)
    OpenGL.GL.glClearColor(0.0, 0.0, 0.0, 1.0)
    self.d_cam.Activate(self.s_cam)
    OpenGL.GL.glPointSize(5) 
    OpenGL.GL.glColor3f(0.0, 1.0,0.0)
    pango.glDrawPoints(kps)
    pango.FinishFrame()


  def addtoqueue(self,kp, pose):
    pose = np.array(pose)
    kp = np.array(kp)
    self.q.put((pose, kp))

  def draw_kp(self,kps):
    OpenGL.GL.glBegin(OpenGL.GL.GL_POINTS)

    for kp in kps:
      OpenGL.GL.glVertex3f(kp[0], kp[1], kp[2])

    OpenGL.GL.glEnd()

  def frame(self):
    if self.d_cam is None:
      return
    while not pango.ShouldQuit():
      draw_kp(self.kps)
      pango.FinishFrame()





