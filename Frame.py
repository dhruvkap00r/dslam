#!/usr/bin/env python3.10
import pypangolin as pango
from OpenGL.GL import *


class Display:
  def __init__(self, pose):
    self.Height = 640
    self.Width = 480
    self.pose = pose
    self.win = pango.CreateWindowAndBind("projection", self.Height, self.Width)
    glEnable(GL_DEPTH_TEST)

    #for both projectionmatrix and modelviewat i need insentric matrix
    #size, focal-point, principle axis, Znear, Zfar

    pm = pango.ProjectionMatrix(self.Width, self.Height, 
                                    420, 420, 
                                    self.Width//2, self.Height//2,
                                    0.1, 1000)
    #camera position and look-at points
    T = None
    mv = pango.ModelViewLookAt(self.pose, (0,0,0), pango.AxisY)
    
    self.s_cam = pango.OpenGlRenderState(pm, mv)

    handler = pango.Handler3D(s_cam)

    d_cam = (pango.CreateDisplay()
              .SetBounds(
                      pango.Attach(0),
                      pango.Attach(1),
                      pango.Attach.Pix(180),
                      pango.Attach(1),
                      -(self.Height / self.Width)
               )
               .SetHandler(handler)
             )



  def frame(self,matches):

    while not pango.ShouldQuit():
      pass        
