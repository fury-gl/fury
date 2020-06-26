
"""
=====================
Simple picking
=====================

Here we present a tutorial of picking objects in the 3D world.

"""

import numpy as np
from fury import actor, window, ui, utils, pick

centers = 0.5 * np.array([[0, 0, 0], [100, 0, 0], [200, 0, 0.]])

colors = np.array([[0.8, 0, 0], [0, 0.8, 0], [0, 0, 0.8]])

radii = 0.1 * np.array([25, 50, 100.])

selected = np.zeros(3, dtype=np.bool)

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
sphere_actor = actor.sphere(centers=centers,
                            colors=colors,
                            radii=radii)

vertices = utils.vertices_from_actor(sphere_actor)
num_vertices = vertices.shape[0]
num_objects = centers.shape[0]


vcolors = utils.colors_from_actor(sphere_actor, 'colors')
print(vcolors.max(), vcolors.min())
print(vcolors.shape)
print(vcolors.dtype)

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

    vertex_index = picked_info['vertex']
    object_index = np.int(np.floor((vertex_index / num_vertices) * num_objects))

    sec = np.int(num_vertices / num_objects)

    if not selected[object_index]:
        scale = 6/5
        color_add = np.array([30, 30, 0], dtype='uint8')
        selected[object_index] = True
    else:
        scale = 5/6
        color_add = np.array([-30, -30, 0], dtype='uint8')
        selected[object_index] = False

    vertices[object_index * sec: object_index * sec + sec] = scale * \
        (vertices[object_index * sec: object_index * sec + sec] -
         centers[object_index]) + centers[object_index]

    vcolors[object_index * sec: object_index * sec + sec] += color_add
    utils.update_actor(sphere_actor)

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
