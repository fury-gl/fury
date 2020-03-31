import numpy as np
from fury import actor, window, ui, utils


fdata = 'xyzr.npy'

#xyzr = np.load(fdata)[:3000, :]
xyzr = np.array([[0, 0, 0, 10], [100, 0, 0, 50], [200, 0, 0, 100]])

colors = np.random.rand(*(xyzr.shape[0], 3))

#colors = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.4], [0, 0, 1., 0.99]])

global text_block
text_block = ui.TextBlock2D(font_size=20, bold=True)
text_block.message = ''
text_block.color = (1, 1, 1)

# panel = ui.Panel2D(position=(150, 90), size=(250, 100),
#                    color=(.6, .6, .6), align="left")
# panel.add_element(text_block, 'relative', (0.2, 0.3))

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

global picker
picker = window.vtk.vtkCellPicker()


def left_click_callback(obj, event):

    global text_block, showm, picker
    x, y, z = obj.GetCenter()
    event_pos = showm.iren.GetEventPosition()

    picker.Pick(event_pos[0], event_pos[1],
                0, showm.scene)

    cell_index = picker.GetCellId()
    point_index = picker.GetPointId()
    text = 'Object ' + str(np.floor((point_index/num_vertices) * num_objects)) + ' Face ID ' + str(cell_index) + '\n' + 'Point ID ' + str(point_index)
    # text_block.message = text
    print(text)
    showm.render()

sphere_actor.AddObserver('LeftButtonPressEvent', left_click_callback, 1)

showm = window.ShowManager(scene, size=(1024, 768), order_transparent=True)
# showm.iren.add_callback()

showm.initialize()
# scene.add(text_block)

#renderer.add(panel)
showm.start()