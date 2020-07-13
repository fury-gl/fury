from fury import actor, window
import numpy as np

scene = window.Scene()
scene.background((1.0, 0.8, 0.8))
dirs = np.random.rand(7, 3)
colors = np.random.rand(7, 3) * 255
centers = np.array([[3, 0, 0], [0, 0, 0], [-3, 0, 0], [0, 3, 0], [0, -3, 0],
                    [0, 0, -3], [0, 0, 3]])

scale = [2, 0.5, 0.5, 2, 1, 1, 1.5]
sdfactor = actor.sdf(centers=centers, directions=dirs, colors=colors,
                     primitives=['torus', 'sphere', 'torus', 'sphere',
                     'sphere', 'torus', 'sphere'],scale=scale)
scene.add(sdfactor)

scene.add(actor.axes())
window.show(scene, size=(1920, 1080))
