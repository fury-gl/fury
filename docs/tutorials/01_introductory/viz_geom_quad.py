
import numpy as np
from fury import window, actor


# centers = np.random.random([6000, 3]) * 300
centers = np.array([[50, 50, 50], [0, 0, 75]])
# colors = np.random.random([6000, 3])
colors = np.random.random([2, 3])

scene = window.Scene()
showm = window.ShowManager(scene, size=(1000, 768))

geom_squares = actor.geom_quad(centers, colors=colors)


squares = actor.square(centers, colors=colors)
# geom_a = actor.geom_actor(centers, colors)
# prim_sphere_actor2 = actor.sphere(colors, colors=colors, radii=0.1)
scene.add(geom_squares, squares)


print(geom_squares)
showm.start()