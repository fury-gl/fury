import copy
import numpy as np
from fury import window, utils, actor, transform
from fury.gltf import glTF
from fury.data import fetch_gltf, read_viz_gltf
from fury.lib import Transform

scene = window.Scene()

fetch_gltf('RiggedFigure', 'glTF')
filename = read_viz_gltf('RiggedFigure')

gltf_obj = glTF(filename, apply_normals=False)
actors = gltf_obj.actors()

# Setting custom opacity to see the bones
# for act in actors:
#     act.GetProperty().SetOpacity(0.7)

vertices = utils.vertices_from_actor(actors[0])
clone = np.copy(vertices)

timeline = gltf_obj.get_skin_timeline()
timeline.add_actor(actors)

scene = window.Scene()
showm = window.ShowManager(scene, size=(900, 768), reset_camera=True,
                           order_transparent=True)
showm.initialize()

bactors = gltf_obj.get_joint_actors(length=0.2, with_transforms=False)
bverts = {}
for bone, joint_actor in bactors.items():
    bverts[bone] = utils.vertices_from_actor(joint_actor)

bvert_copy = copy.deepcopy(bverts)

scene.add(timeline)
scene.add(* bactors.values())

bones = gltf_obj.bones
parent_transforms = gltf_obj.bone_tranforms


def transverse_timelines(timeline, bone_id, timestamp, joint_matrices,
                         parent_bone_deform=np.identity(4)):
    deform = timeline.get_value('transform', timestamp)
    new_deform = np.dot(parent_bone_deform, deform)

    # calculating skinning metrix
    ibm = gltf_obj.ibms[bone_id].T
    skin_matrix = np.dot(new_deform, ibm)
    joint_matrices[bone_id] = skin_matrix

    node = gltf_obj.gltf.nodes[bone_id]
    actor_transform = gltf_obj.transformations[0]
    bone_transform = np.dot(actor_transform, new_deform)
    bverts[bone_id][:] = transform.apply_transfomation(bvert_copy[bone_id],
                                                       bone_transform)
    utils.update_actor(bactors[bone_id])
    if node.children:
        for c_timeline, c_bone in zip(timeline.timelines, node.children):
            transverse_timelines(c_timeline, c_bone, timestamp,
                                 joint_matrices, new_deform)


def timer_callback(_obj, _event):
    timeline.update_animation()
    timestamp = timeline.current_timestamp
    joint_matrices = {}

    for child in timeline.timelines:
        transverse_timelines(child, bones[0], timestamp, joint_matrices)

    vertices[:] = gltf_obj.apply_skin_matrix(clone, joint_matrices)
    utils.update_actor(actors[0])
    utils.compute_bounds(actors[0])
    showm.render()


showm.add_timer_callback(True, 10, timer_callback)

showm.start()
