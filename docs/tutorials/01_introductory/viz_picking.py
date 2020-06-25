
"""
=====================
Simple picking
=====================

Here we present a tutorial of picking objects in the 3D world.

"""

import numpy as np
from fury import actor, window, ui, utils, pick

xyzr = np.array([[0, 0, 0, 25], [100, 0, 0, 50], [200, 0, 0, 100]])

colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.]])

###############################################################################
# Let's create a panel to show what is picked

panel = ui.Panel2D(size=(400, 150), color=(1, .5, .0), align="right")
panel.center = (150, 200)

text_block = ui.TextBlock2D(text="Pick a \nsphere")
panel.add_element(text_block, (0.3, 0.3))

###############################################################################
# Build scene and add an actor with many objects.

scene = window.Scene()

label_actor = actor.label(text='Test')
sphere_actor = actor.sphere(centers=0.5 * xyzr[:, :3],
                            colors=colors[:],
                            radii=0.1 * xyzr[:, 3])

vertices = utils.vertices_from_actor(sphere_actor)
num_vertices = vertices.shape[0]
num_objects = xyzr.shape[0]

ax = actor.axes(scale=(10, 10, 10))

scene.add(sphere_actor)
scene.add(label_actor)
scene.add(ax)
scene.reset_camera()

global showm

###############################################################################
# Select a picking option


pickm = pick.PickingManager()


def left_click_callback(obj, event):

    global text_block, showm

    event_pos = pickm.event_position(showm.iren)

    picked_info = pickm.pick(event_pos[0], event_pos[1],
                             0, showm.scene)
    print(picked_info)

    try:
        vertex_index = picked_info['vertex']
        object_index = np.floor((vertex_index / num_vertices) * num_objects)
    except TypeError:
        object_index = None

    face_index = picked_info['face']

    text = 'Object ' + str(object_index) + '\n'
    text += 'Vertex ID ' + str(vertex_index) + '\n'
    text += 'Face ID ' + str(face_index) + '\n'
    text += 'World pos ' + str(np.round(picked_info['xyz'], 2))
    text_block.message = text
    showm.render()


sphere_actor.AddObserver('LeftButtonPressEvent', left_click_callback, 1)

showm = window.ShowManager(scene, size=(1024, 768), order_transparent=True)

showm.initialize()
scene.add(panel)
showm.start()
