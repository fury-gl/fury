import numpy as np
from fury import window, actor

centers = np.random.random([500, 3]) * 100
colors = np.random.random([500, 3])

scene = window.Scene()
showm = window.ShowManager(scene, size=(1000, 768))

geom_squares = actor.billboard_gs(centers, colors=colors)
scene.add(geom_squares)

showm.start()
