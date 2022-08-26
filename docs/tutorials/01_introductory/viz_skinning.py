import numpy as np
from fury import window, utils, actor, transform
from fury.gltf import glTF
from fury.data import fetch_gltf, read_viz_gltf
from fury.lib import Transform

scene = window.Scene()

fetch_gltf('SimpleSkin', 'glTF')
filename = read_viz_gltf('RiggedSimple')

gltf_obj = glTF(filename, apply_normals=False)
actors = gltf_obj.actors()

vertices = utils.vertices_from_actor(actors[0])
clone = np.copy(vertices)

timelines = gltf_obj.get_skin_timelines()
timelines.add_actor(actors[0])

scene = window.Scene()
showm = window.ShowManager(scene, size=(900, 768), reset_camera=False,
                           order_transparent=True)
showm.initialize()

scene.add(timelines)

bones = gltf_obj.bones[0]


def timer_callback(_obj, _event):
    timelines.update_animation()
    joint_matrices = []
    ibms = []
    for i, bone in enumerate(bones):
        if timelines.is_interpolatable(f'transform{i}'):
            deform = timelines.get_value(f'transform{i}',
                                         timelines.current_timestamp)
            ibm = gltf_obj.ibms[0][i].T
            ibms.append(ibm)
            local_transform = gltf_obj.bone_tranforms[bone]
            joint_matrices.append(deform)

    vertices[:] = gltf_obj.apply_skin_matrix(clone, joint_matrices,
                                             bones, ibms)
    utils.update_actor(actors[0])
    utils.compute_bounds(actors[0])
    showm.render()


showm.add_timer_callback(True, 10, timer_callback)

showm.start()
