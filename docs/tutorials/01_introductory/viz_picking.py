
"""
=====================
Simple picking
=====================

Here we present a tutorial of picking objects in the 3D world.

"""

import numpy as np
from fury import actor, window, ui, utils

xyzr = np.array([[0, 0, 0, 25], [100, 0, 0, 50], [200, 0, 0, 100]])

colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.]])

###############################################################################
# Let's create a panel to show what is picked

panel = ui.Panel2D(size=(200, 150), color=(1, .5, .0), align="right")
panel.center = (150, 200)

text_block = ui.TextBlock2D(text="Pick a \nsphere")
panel.add_element(text_block, (0.3, 0.3))

###############################################################################
# Build scene and add an actor with many objects.

scene = window.Scene()

sphere_actor = actor.sphere(centers=0.5 * xyzr[:, :3],
                            colors=colors[:],
                            radii=0.1 * xyzr[:, 3])

vertices = utils.vertices_from_actor(sphere_actor)
num_vertices = vertices.shape[0]
num_objects = xyzr.shape[0]

scene.add(sphere_actor)
scene.reset_camera()

global showm

def picking(mode='face'):
    if mode == 'face':
        picker = window.vtk.vtkCellPicker()
    elif mode == 'vertex':
        picker = window.vtk.vtkPointPicker()
    else:
        raise ValueError('Unknown picker option')

    def pick(x, y, z, sc):
        picker.Pick(x, y, z, sc)
        if mode == 'face':
            return {'vertex': picker.GetPointId(),
                    'face' : picker.GetCellId()}
        if mode == 'vertex':
            return {'vertex': picker.GetPointId(),
                    'face' : None}

    picker.pick = pick
    return picker


###############################################################################
# Select a picking option

global picker
picker = picking('face')
# picker = picking('vertex')


def left_click_callback(obj, event):

    global text_block, showm, picker
    event_pos = showm.iren.GetEventPosition()

    picked_info = picker.pick(event_pos[0], event_pos[1],
                              0, showm.scene)
    print(picked_info)
    vertex_index = picked_info['vertex']
    face_index = picked_info['face']
    object_index = np.floor((vertex_index/num_vertices) * num_objects)
    text = 'Object ' + str(object_index) + '\n'
    text += 'Vertex ID ' + str(vertex_index) + '\n'
    text += 'Face ID ' + str(face_index)
    text_block.message = text
    showm.render()


sphere_actor.AddObserver('LeftButtonPressEvent', left_click_callback, 1)

showm = window.ShowManager(scene, size=(1024, 768), order_transparent=True)

showm.initialize()
scene.add(panel)
showm.start()
