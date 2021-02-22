from fury.tests import test_actors as ta
import numpy as np
from fury import window, actor
import itertools


dirs = np.array([[0.01365219, 0.47699548, 0.99209483],
 [0.56080887, 0.88879732, 0.29196388],
 [0.83677901, 0.00300227, 0.11017777]])


# dirs = np.random.rand(3, 3)
colors = np.random.rand(3, 3) * 255
centers = np.array([[1, 0, 0], [0, 0, 0], [-1, 0, 0]])
scales = np.random.rand(3, 1)


# dirs = np.random.rand(1, 3)
# colors = np.random.rand(1, 3) * 255
# centers = np.array([[0, 0, 0]])
# scales = np.random.rand(1, 1)


# sdfactor = actor.sdf(centers=centers, directions=dirs, colors=colors,
#                      primitives=['sphere', 'torus', 'superellipse'],
#                      scales=scales)

# sdfactor = actor.sdf(centers=centers, directions=dirs, colors=colors,
#                      primitives=['superellipse', 'sphere', 'superellipse'],
#                      scales=scales)

# sdfactor = actor.sdf(centers=centers, directions=dirs, colors=colors,
                    #  primitives=['sphere', 'torus', 'ellipsoid'],
                    #  scales=scales)

# sdfactor = actor.sdf(centers=centers, directions=dirs, colors=colors,
#                      primitives=['superellipse'],
#                      scales=scales)

sdfactor = actor.sdf(centers=centers, directions=dirs, colors=colors,
                     primitives=['superellipse', 'superellipse', 'superellipse'],
                     scales=scales)

# sdfactor = actor.sdf(centers=centers, directions=dirs, colors=colors,
#                      primitives=['superellipse'],
#                      scales=scales)


scene = window.Scene()
scene.background((1.0, 0.8, 0.8))
scene.add(sdfactor)


current_size = (1024, 720)
showm = window.ShowManager(scene, size=current_size,
                           title="Visualize SDF Actor")

counter = itertools.count()
print(dirs)

# ta.test_sdf_actor()

def timer_callback(_obj, _event):
    cnt = next(counter)
    showm.scene.azimuth(cnt * 0.05)
    showm.render()
    if cnt == 20000:
        showm.exit()


showm.initialize()
showm.add_timer_callback(True, 100, timer_callback)
showm.start()


# [[0.01365219 0.47699548 0.99209483]
#  [0.56080887 0.88879732 0.29196388]
#  [0.83677901 0.00300227 0.11017777]]
