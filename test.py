import numpy as np
from fury import window, actor

dirs = np.random.rand(4, 3)
colors = np.random.rand(4, 3) * 255
centers = np.array([[1, 0, 0], [0, 0, 0], [-1, 0, 0], [0, 1, 0]]) * 10
# scales = np.random.rand(4, 1)
# scales = np.array([1, 1, 10, 10])
scales = np.array([100, 1, 10, 10])

# sdfactor = actor.sdf(centers=centers, directions=dirs, colors=colors,
#                      primitives=['sphere', 'torus', 'superellipsoid', 'capsule'],
#                      scales=scales)


sdfactor = actor.sdf(centers=centers, directions=dirs, colors=colors,
                     primitives=['sphere', 'torus', 'sphere', 'torus'],
                     scales=scales)

scene = window.Scene()
scene.background((1.0, 0.8, 0.8))
scene.add(sdfactor)


current_size = (1024, 720)
showm = window.ShowManager(scene, size=current_size,
                           title="Visualize SDF Actor")

interactive = True

if interactive:
    showm.start()

window.record(scene, out_path='viz_sdfactor.png', size=current_size)
