import numpy as np
from fury import window, actor


centers = np.random.random([30, 3]) * 20
colors = np.random.random([30, 3])

# creating a scene
scene = window.Scene()

geom_squares = actor.billboard_gs(centers, colors=colors)
scene.add(geom_squares)


interactive = False
if interactive:
    window.show(scene)

window.record(scene, size=(600, 600), out_path="viz_billboard_geom_shader.png")
