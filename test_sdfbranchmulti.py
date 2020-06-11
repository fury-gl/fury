from fury import actor, window
import numpy as np

scene = window.Scene()
scene.background((1.0, 0.8, 0.8))
centers = np.array([[0, 0, 0]])
sdfactor = actor.multi_sdf(centers=centers, scale=6)
scene.add(sdfactor)

scene.add(actor.axes())
window.show(scene, size=(1920, 1080))
