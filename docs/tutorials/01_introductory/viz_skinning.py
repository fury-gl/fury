import numpy as np
from fury import window, utils, actor
from fury.gltf import glTF
from fury.data import fetch_gltf, read_viz_gltf
from fury.lib import Transform

scene = window.Scene()

fetch_gltf('SimpleSkin', 'glTF')
filename = read_viz_gltf('SimpleSkin')

gltf_obj = glTF(filename, apply_normals=True)
actors = gltf_obj.actors()
vertices = utils.vertices_from_actor(actors[0])
clone = np.copy(vertices)
# timeline = gltf_obj.get_skin_timeline()
timelines = gltf_obj.get_skin_timelines()[0]
timelines.add_actor(actors[0])
print(len(timelines))

scene = window.Scene()
showm = window.ShowManager(scene, size=(900, 768), reset_camera=False,
                           order_transparent=True)
showm.initialize()

scene.add(timelines)

print(f'vertices: {vertices}')
print(f'weights: {gltf_obj.weights_0}')


def timer_callback(_obj, _event):
    timelines.update_animation()
    # if timelines[0].is_interpolatable('transform'):
    deform = timelines.get_value('transform', timelines.current_timestamp)
    # print(deform)
    vertices[:] = gltf_obj.apply_skin_matrix(clone, deform)
    utils.update_actor(actors[0])
    showm.render()


showm.add_timer_callback(True, 10, timer_callback)

showm.start()
