import numpy as np

from fury import window, utils
from fury.io import save_polydata, load_polydata
from fury.utils import vtk

my_polydata = vtk.vtkPolyData()

my_vertices = np.array([[1.0, 0.0, 0.0], #0
                       [2.0, 0.75, 0.0], #1
                       [3.0, 0.0, 0.0], #2
                       [2.75, 1.25, 0.0], #3
                       [3.75, 2.0, 0.0], #4
                       [2.5, 2.0, 0.0], #5
                       [2.0, 3.0, 0.0], #6
                       [1.5, 2.0, 0.0], #7
                       [0.25, 2.0, 0.0], #8
                       [1.25, 1.25, 0.0]]) #9

my_triangles = np.array([[1, 9, 0], #good
                         [1, 2, 3],
                         [3, 4, 5], #good
                         [5, 6, 7],
                         [7, 8, 9], #good
                         [1, 9, 3], #good
                         [3, 7, 9], #good
                         [3, 5, 7]], dtype='i8') #good

utils.set_polydata_vertices(my_polydata, my_vertices)
utils.set_polydata_triangles(my_polydata, my_triangles)

file_name = "my_star2D.vtk"
save_polydata(my_polydata, file_name)
print("Surface saved in " + file_name)

star_polydata = load_polydata(file_name)

star_vertices = utils.get_polydata_vertices(star_polydata)
colors = star_vertices * 255
utils.set_polydata_colors(star_polydata, colors)

print("new surface colors")
print(utils.get_polydata_colors(star_polydata))

# get vtkActor
star_actor = utils.get_actor_from_polydata(star_polydata)
star_actor.GetProperty().BackfaceCullingOff()
# Create a scene
scene = window.Scene()
scene.add(star_actor)
scene.set_camera(position=(0, 0, 7), focal_point=(0, 0, 0))
scene.zoom(3)

# display
# window.show(scene, size=(1000, 1000), reset_camera=False) this allows the picture to be moved around
window.record(scene, out_path='star2D.png', size=(600, 600))
