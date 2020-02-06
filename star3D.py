import numpy as np

from fury import window, utils
from fury.io import save_polydata, load_polydata
from fury.utils import vtk

my_polydata = vtk.vtkPolyData()

my_vertices = np.array([[1.0, 0.0, 1.0], #0
                       [2.0, 0.75, 1.0], #1
                       [3.0, 0.0, 1.0], #2
                       [2.75, 1.25, 1.0], #3
                       [3.75, 2.0, 1.0], #4
                       [2.5, 2.0, 1.0], #5
                       [2.0, 3.0, 1.0], #6
                       [1.5, 2.0, 1.0], #7
                       [0.25, 2.0, 1.0], #8
                       [1.25, 1.25, 1.0], #9
                       [2.0, 1.0, 1.5], #10
                       [2.0, 1.0, 0.5]]) #11


my_triangles = np.array([[1, 9, 0], #2D section
                         [1, 2, 3],
                         [3, 4, 5],
                         [5, 6, 7],
                         [7, 8, 9],
                         [1, 9, 3],
                         [3, 7, 9],
                         [3, 5, 7],
                         [1, 0, 10], #start of 3D section, front, change all 10's to 11 for back of star3D
                         [0, 9, 10],
                         [10, 9, 8],
                         [7, 8, 10],
                         [6, 7, 10],
                         [5, 6, 10],
                         [5, 10, 4],
                         [10, 3, 4],
                         [3, 10, 2],
                         [10, 1, 2], #end of front
                         [1, 0, 11], #Start of Back
                         [0, 9, 11],
                         [11, 9, 8],
                         [7, 8, 11],
                         [6, 7, 11],
                         [5, 6, 11],
                         [5, 11, 4],
                         [11, 3, 4],
                         [3, 11, 2],
                         [11, 1, 2]], dtype='i8') #good

utils.set_polydata_vertices(my_polydata, my_vertices)
utils.set_polydata_triangles(my_polydata, my_triangles)

file_name = "my_star3D.vtk"
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
star_actor.GetProperty().BackfaceCullingOff() #gets rid of the winding order issue (look at later and other algorithms that get rid of winding order)

# Create a scene
scene = window.Scene()
scene.add(star_actor)
scene.set_camera(position=(0, 0, 7), focal_point=(0, 0, 0))
scene.zoom(0)

# display
window.show(scene, size=(1000, 1000), reset_camera=False) #this allows the picture to be moved around
window.record(scene, out_path='star3D.png', size=(600, 600))
