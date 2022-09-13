import numpy as np
from fury import window, utils, actor, transform
from fury.gltf import glTF
from fury.data import fetch_gltf, read_viz_gltf
from fury.lib import Transform

scene = window.Scene()

fetch_gltf('SimpleSkin', 'glTF')
filename = read_viz_gltf('RiggedFigure')

gltf_obj = glTF(filename, apply_normals=False)
actors = gltf_obj.actors()
print(len(actors))

vertices = utils.vertices_from_actor(actors[0])
clone = np.copy(vertices)

timeline = gltf_obj.get_skin_timeline2()
timeline.add_actor(actors[0])

scene = window.Scene()
showm = window.ShowManager(scene, size=(900, 768), reset_camera=False,
                           order_transparent=True)
showm.initialize()

bactors = gltf_obj.get_joint_actors(length=0.2)
bverts = []
for bone, joint_actor in bactors.items():
    bverts.append(utils.vertices_from_actor(joint_actor))

bvert_copy = np.copy(bverts)

scene.add(* bactors.values())
scene.add(timeline)

bones = gltf_obj.bones[0]
parent_transforms = gltf_obj.bone_tranforms


def timer_callback(_obj, _event):
    timeline.update_animation()
    joint_matrices = []
    ibms = []

    for i, bone in enumerate(bones):
        if timeline.is_interpolatable(f'transform{bone}'):
            deform = timeline.get_value(f'transform{bone}',
                                        timeline.current_timestamp)
            ibm = gltf_obj.ibms[0][i].T
            ibms.append(ibm)

            parent_transform = parent_transforms[bone]
            joint_mat = np.dot(parent_transform, deform)
            joint_mat = np.dot(joint_mat, ibm)
            joint_matrices.append(joint_mat)

            # if bone == 12:
            bverts[i][:] = transform.apply_transfomation(bvert_copy[i], deform)
            utils.update_actor(bactors[bone])

    vertices[:] = gltf_obj.apply_skin_matrix(clone, joint_matrices,
                                             bones, ibms)
    utils.update_actor(actors[0])
    utils.compute_bounds(actors[0])
    showm.render()


showm.add_timer_callback(True, 10, timer_callback)

showm.start()
