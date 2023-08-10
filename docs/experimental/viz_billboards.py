import numpy as np

from fury.actor import billboard
from fury.window import Scene, ShowManager

width, height = (1350, 800)
cam_pos = np.array([0.0, 0.0, -1.0])

scene = Scene()
scene.set_camera(position=(cam_pos[0], cam_pos[1], cam_pos[2]),
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

center = np.array([[0.0, 0.0, 20.0]])


#hardcoding the billboard to be fullscreen and centered
scale_factor_2 = np.abs(np.linalg.norm(center[0] - cam_pos))*np.sin(np.deg2rad(scene.camera().GetViewAngle()/2.0))
print(scale_factor_2)

scale = scale_factor_2*np.array([[width/height, 1.0, 0.0]])

bill = billboard(center, scales=scale, colors = (1.0, 0.0, 0.0))

def callback(obj = None, event = None):
    pos, fp, vu = manager.scene.get_camera()
    scale_factor_2 = np.abs(np.linalg.norm(center[0] - np.array(pos)))*np.sin(np.deg2rad(scene.camera().GetViewAngle()/2.0))
    scale = scale_factor_2*np.array([[width/height, 1.0, 0.0]])
    bill.SetScale(scale[0, 0], scale[0, 1], scale[0, 2])
    bill.Modified()

callback()

manager.scene.add(bill)
manager.add_iren_callback(callback, "RenderEvent")

manager.start()