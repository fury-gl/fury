import copy
import numpy as np
from fury import window, transform
from fury.utils import vertices_from_actor, update_actor, compute_bounds
from fury.gltf import glTF
from fury.data import fetch_gltf, read_viz_gltf

scene = window.Scene()
show_bones = False

fetch_gltf('RiggedFigure', 'glTF')
filename = read_viz_gltf('RiggedFigure')

gltf_obj = glTF(filename, apply_normals=False)
actors = gltf_obj.actors()

vertices = [vertices_from_actor(actor) for actor in actors]
clone = [np.copy(vert) for vert in vertices]

timeline = gltf_obj.get_skin_timeline()['anim_0']
timeline.add_actor(actors)

scene = window.Scene()
showm = window.ShowManager(scene, size=(900, 768), reset_camera=True,
                           order_transparent=True)
showm.initialize()
scene.add(timeline)

if show_bones:
    bactors = gltf_obj.get_joint_actors(length=0.2, with_transforms=False)
    bverts = {}
    for bone, joint_actor in bactors.items():
        bverts[bone] = vertices_from_actor(joint_actor)
    bvert_copy = copy.deepcopy(bverts)
    scene.add(* bactors.values())

bones = gltf_obj.bones
parent_transforms = gltf_obj.bone_tranforms


def transverse_timelines(timeline, bone_id, timestamp, joint_matrices,
                         parent_bone_deform=np.identity(4)):
    deform = timeline.get_value('transform', timestamp)
    new_deform = np.dot(parent_bone_deform, deform)

    # calculating skinning matrix
    ibm = gltf_obj.ibms[bone_id].T
    skin_matrix = np.dot(new_deform, ibm)
    joint_matrices[bone_id] = skin_matrix

    node = gltf_obj.gltf.nodes[bone_id]

    if show_bones:
        actor_transform = gltf_obj.transformations[0]
        bone_transform = np.dot(actor_transform, new_deform)
        bverts[bone_id][:] = transform.apply_transfomation(bvert_copy[bone_id],
                                                           bone_transform)
        update_actor(bactors[bone_id])

    if node.children:
        c_timelines = timeline.timelines
        c_bones = node.children
        for c_timeline, c_bone in zip(c_timelines, c_bones):
            transverse_timelines(c_timeline, c_bone, timestamp,
                                 joint_matrices, new_deform)


def timer_callback(_obj, _event):
    timeline.update_animation()
    timestamp = timeline.current_timestamp
    joint_matrices = {}
    root_bone = gltf_obj.gltf.skins[0].skeleton
    root_bone = root_bone if root_bone else gltf_obj.bones[0]

    if not root_bone == bones[0]:
        _timeline = timeline.timelines[0]
        parent_transform = gltf_obj.transformations[root_bone].T
    else:
        _timeline = timeline
        parent_transform = np.identity(4)
    for child in _timeline.timelines:
        transverse_timelines(child, bones[0], timestamp,
                             joint_matrices, parent_transform)
    for i, vertex in enumerate(vertices):
        vertex[:] = gltf_obj.apply_skin_matrix(clone[i], joint_matrices, i)
        update_actor(actors[i])
        compute_bounds(actors[i])
    showm.render()


showm.add_timer_callback(True, 20, timer_callback)

showm.start()
