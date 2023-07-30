import numpy as np

from fury.actor import billboard
from fury.window import Scene, ShowManager

width, height = (1350, 800)

scene = Scene()
scene.set_camera(position=(-6, 5, -10),
                 focal_point=(0.0,
                              0.0,
                              0.0),
                 view_up=(0.0, 0.0, 0.0))

manager = ShowManager(
    scene,
    "demo",
    (width,
     height))

manager.initialize()

scale = 3.4*np.array([[width/height, 1.0, 0.0]])

bill = billboard(np.array([[0.0, 0.0, 0.0]]), scales=scale,colors = (1.0, 0.0, 0.0))
manager.scene.add(bill)

manager.start()