import numpy as np
from fury import actor, gltf, window

centers = np.zeros((3, 3))
directions = np.array([1, 1, 0])
colors = np.array([1, 1, 1])

cube = actor.cube(centers, colors=colors)

scene = window.Scene()
scene.add(cube)

cone = actor.sphere(np.add(centers, np.array([2, 0, 0])),
                    colors)
scene.add(cone)
scene.set_camera(position=(4.45, -21, 12), focal_point=(4.45, 0.0, 0.0),
                 view_up=(0.0, 0.0, 1.0))
gltf.export_scene(scene, filename="viz_gltf.glb")

scene.clear()

# Reading model
gltf_obj = gltf.glTF('viz_glb.glb')
actors = gltf_obj.get_actors()  
scene.add(*actors)
window.show(scene)
