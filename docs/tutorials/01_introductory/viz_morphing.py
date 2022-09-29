import copy
import numpy as np
from fury import window
from fury.utils import vertices_from_actor, update_actor, compute_bounds
from fury.gltf import glTF
from fury.data import fetch_gltf, read_viz_gltf

scene = window.Scene()

fetch_gltf('AnimatedMorphCube', 'glTF')
filename = read_viz_gltf('AnimatedMorphCube')

gltf_obj = glTF(filename, apply_normals=False)
actors = gltf_obj.actors()

vertices = [vertices_from_actor(actor) for actor in actors]
clone = [np.copy(vert) for vert in vertices]

timeline = gltf_obj.morph_timeline()['Square']
timeline.add_actor(actors)

scene = window.Scene()
showm = window.ShowManager(scene, size=(900, 768), reset_camera=True,
                           order_transparent=True)
showm.initialize()
scene.add(timeline)


def timer_callback(_obj, _event):
    timeline.update_animation()
    timestamp = timeline.current_timestamp
    for i, verts in enumerate(vertices):
        weights = timeline.timelines[0].get_value('morph', timestamp)
        verts[:] = gltf_obj.apply_morph_vertices(clone[i], weights, i)
        update_actor(actors[i])
        compute_bounds(actors[i])
    showm.render()


showm.add_timer_callback(True, 20, timer_callback)
showm.start()
