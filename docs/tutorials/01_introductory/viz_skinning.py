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
print(vertices)
# timeline = gltf_obj.get_skin_timeline()
timelines = gltf_obj.get_skin_timelines()
timelines.add_actor(actors[0])
# print(len(timelines))

scene = window.Scene()
showm = window.ShowManager(scene, size=(900, 768), reset_camera=False,
                           order_transparent=True)
showm.initialize()

scene.add(timelines)

bones = gltf_obj.bones[0]
# print(f'bones: {bones}')
# print(f'ibms: {gltf_obj.ibms}')


def timer_callback(_obj, _event):
    timelines.update_animation()
    joint_matrices = []
    for i, bone in enumerate(bones):
        if timelines.is_interpolatable(f'transform{i}'):
            deform = timelines.get_value(f'transform{i}', timelines.current_timestamp)
            ibm = gltf_obj.ibms[0][i]
            ibm = np.linalg.inv(ibm.T)
            deform = np.dot(deform, ibm)
            joint_matrices.append(deform)
    # print(clone)
    vertices[:] = gltf_obj.apply_skin_matrix(clone, joint_matrices, bones)
    # print(vertices)
    utils.update_actor(actors[0])
    showm.render()


showm.add_timer_callback(True, 10, timer_callback)

showm.start()
