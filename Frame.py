#!/usr/bin/env python3.10
import pypangolin as pango
import OpenGL.GL as gl


class Display:
  def __init__(self):
    self.Height = 640
    self.Width = 480
    self.win = pangolin.CreateWindowAndBind("projection", self.Height, self.Width)
    glEnable(GL_DEPTH_TEST)

    #for both projectionmatrix and modelviewat i need insentric matrix
    #size, focal-point, principle axis, Znear, Zfar

    pm = pango.ProjectionMatrix(self.Width, self.Height, 
                                    420, 420, 
                                    self.Width//2, self.Height//2,
                                    0.1, 1000)
    #camera position and look-at points

    mv = pango.ModelViewLookAt(#camera postiion, look-at point, pango.AxisY)
    
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
        
dis = Display()
example_matches = [[(100, 100)], [(300, 200)], [(500, 400)]]
dis.frame(example_matches)
