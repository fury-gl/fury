import numpy as np
from fury import actor, window

scene = window.Scene()
g_actor = actor.pentagonalprism(centers=np.array([[0, 0, 0]]))  # actor.triangularprism(centers=np.array([[0, 0, 0]]))
scene.add(g_actor, actor.axes())
window.show(scene)
