
"""
==========================
Selecting multiple objects
==========================

Here we present a tutorial showing how to interact with objects in the
3D world. All objects to be picked are part of a single actor.
FURY likes to bundle objects in a few actors to reduce code and
increase speed.

When the objects will be picked they will change size and color.
"""

import numpy as np
from fury import actor, window, ui, utils, pick

centers = 0.5 * np.array([[0, 0, 0], [100, 0, 0], [200, 0, 0.]])
colors = np.array([[0.8, 0, 0], [0, 0.8, 0], [0, 0, 0.8]])
radii = 0.1 * np.array([50, 100, 150.])

num_faces = 3 * 6 * 2  # every quad of each cubes has 2 triangles

selected = np.zeros(3, dtype=bool)

###############################################################################
# Let's create a panel to show what is picked

panel = ui.Panel2D(size=(400, 200), color=(1, .5, .0), align="right")
panel.center = (150, 200)

text_block = ui.TextBlock2D(text="Left click on object \n")
panel.add_element(text_block, (0.3, 0.3))

###############################################################################
# Build scene and add an actor with many objects.

scene = window.Scene()

label_actor = actor.label(text='Test')

###############################################################################
# This actor is made with 3 cubes

fury_actor = actor.cube(centers, directions=(1, 0, 0), colors=colors, scales=radii)

###############################################################################
# Access the memory of the vertices of all the cubes

vertices = utils.vertices_from_actor(fury_actor)
num_vertices = vertices.shape[0]
num_objects = centers.shape[0]

###############################################################################
# Access the memory of the colors of all the cubes

vcolors = utils.colors_from_actor(fury_actor, 'colors')

###############################################################################
# Adding an actor showing the axes of the world coordinates
ax = actor.axes(scale=(10, 10, 10))

rgba = 255 * np.ones((100, 200, 4))
rgba[1:-1, 1:-1] = np.zeros((98, 198, 4))
#rgba = np.round(255 * np.random.rand(100, 200, 4), 0)

texa = actor.texture_2d(rgba.astype(np.uint8))

scene.add(fury_actor)
scene.add(label_actor)
scene.add(ax)
scene.add(texa)
scene.reset_camera()


###############################################################################
# Create the Picking manager

selm = pick.SelectionManager(select='faces')

###############################################################################
# Time to make the callback which will be called when we pick an object


def left_click_callback(obj, event):

    # Get the event position on display and pick

    event_pos = selm.event_position(showm.iren)
    picked_info = selm.pick(event_pos, showm.scene)

    print(picked_info)

    face_index = picked_info['face'][0]

    # Calculate the objects index

    object_index = int(np.floor((face_index / 6 * 2)))

    # Find how many vertices correspond to each object
    sec = int(num_vertices / num_objects)

    if not selected[object_index]:
        scale = 6/5
        color_add = np.array([30, 30, 30], dtype='uint8')
        selected[object_index] = True
    else:
        scale = 5/6
        color_add = np.array([-30, -30, -30], dtype='uint8')
        selected[object_index] = False

    # Update vertices positions
    vertices[object_index * sec: object_index * sec + sec] = scale * \
        (vertices[object_index * sec: object_index * sec + sec] -
         centers[object_index]) + centers[object_index]

    # Update colors
    vcolors[object_index * sec: object_index * sec + sec] += color_add

    # Tell actor that memory is modified
    utils.update_actor(fury_actor)

    face_index = picked_info['face']

    # Show some info
    text = 'Object ' + str(object_index) + '\n'
    text += 'Vertex ID ' + str(vertex_index) + '\n'
    text += 'Face ID ' + str(face_index) + '\n'
    text += 'Actor ID ' + str(id(picked_info['actor']))
    text_block.message = text
    showm.render()


def hover_callback(_obj, _event):
    event_pos = selm.event_position(showm.iren)
    info = selm.select(event_pos, showm.scene, (10, 10))
    print(info)
    showm.render()


###############################################################################
# Make the window appear

showm = window.ShowManager(scene, size=(1024, 768), order_transparent=True)
showm.initialize()


###############################################################################
# Bind the callback to the actor
showm.add_iren_callback(hover_callback)

scene.add(panel)

###############################################################################
# Change interactive to True to start interacting with the scene

interactive = True

if interactive:

    showm.start()


###############################################################################
# Save the current framebuffer in a PNG file

window.record(showm.scene, size=(1024, 768), out_path="viz_picking.png")
