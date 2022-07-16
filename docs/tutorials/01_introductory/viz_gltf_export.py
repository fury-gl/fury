import numpy as np
from fury import actor, gltf, window
from fury.data import fetch_gltf, read_viz_gltf

centers = np.zeros((3, 3))
directions = np.array([1, 1, 0])
colors = np.array([1, 1, 1])

cube = actor.cube(np.add(centers, np.array([2, 0, 0])), colors=colors)

scene = window.Scene()
scene.add(cube)

cone = actor.sphere(np.add(centers, np.array([0, 2, 0])),
                    colors)
scene.add(cone)

fetch_gltf('BoxTextured', 'glTF')
filename = read_viz_gltf('BoxTextured')
gltf_obj = gltf.glTF(filename)
box_actor = gltf_obj.actors()
scene.add(box_actor[0])

scene.set_camera(position=(4.45, -21, 12), focal_point=(4.45, 0.0, 0.0),
                 view_up=(0.0, 0.0, 1.0))
gltf.export_scene(scene, filename="viz_gltf_export.gltf")

scene.clear()

# Reading model
gltf_obj = gltf.glTF('viz_gltf_export.gltf')
actors = gltf_obj.actors()
scene.add(*actors)
window.show(scene)
