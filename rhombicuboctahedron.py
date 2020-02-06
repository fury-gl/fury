import numpy as np

from fury import window, utils
from fury.io import save_polydata, load_polydata
from fury.utils import vtk

my_polydata = vtk.vtkPolyData()

my_vertices = np.array([[2.0, 8.0, 6.0], #0
                       [0.0, 6.0, 6.0],#1
                       [0.0, 2.0, 6.0],#2
                       [2.0, 0.0, 6.0],#3
                       [6.0, 0.0, 6.0],#4
                       [8.0, 2.0, 6.0],#5
                       [8.0, 6.0, 6.0],#6
                       [6.0, 8.0, 6.0],#7
                       [2.0, 6.0, 8.0],#8
                       [2.0, 2.0, 8.0],#9
                       [6.0, 6.0, 8.0],#10
                       [6.0, 2.0, 8.0],#11
                       [2.0, 8.0, 2.0],#0 -start of 2nd face
                       [0.0, 6.0, 2.0],#1
                       [0.0, 2.0, 2.0],#2
                       [2.0, 0.0, 2.0],#3
                       [6.0, 0.0, 2.0],#4
                       [8.0, 2.0, 2.0],#5
                       [8.0, 6.0, 2.0],#6
                       [6.0, 8.0, 2.0],#7
                       [2.0, 6.0, 0.0],#8
                       [2.0, 2.0, 0.0],#9
                       [6.0, 6.0, 0.0],#10
                       [6.0, 2.0, 0.0]])#11

my_triangles = np.array([[0, 1, 8], #1
                         [1, 2, 9], #2
                         [1, 8, 9], #3
                         [2, 3, 9], #4
                         [3, 9, 11], #5
                         [3, 4, 11], #6
                         [4, 11, 5], #7
                         [5, 11, 10], #8
                         [5, 10, 6], #9
                         [6, 7, 10], #10
                         [7, 8, 10], #11
                         [7, 8, 0], #12
                         [8, 9, 10], #13
                         [9, 10, 11], #14 end of front face, works up to here
                         [12, 13, 20], #1
                         [13, 14, 21], #2
                         [13, 20, 21], #3
                         [14, 15, 21], #4
                         [15, 21, 23], #5
                         [15, 16, 23], #6
                         [16, 23, 17], #7
                         [17, 22, 23], #8
                         [17, 22, 18], #9
                         [18, 19, 22], #10
                         [19, 20, 22], #11
                         [19, 20, 12], #12
                         [20, 21, 22], #13
                         [21, 22, 23], #14
                         [7, 18, 19], #end of back face, start of right side
                         [6, 7, 18],
                         [6, 17, 18],
                         [5, 6, 17],
                         [4, 5, 16],
                         [5, 16, 17], #end of right side, start of left side
                         [0, 1, 12],
                         [1, 12, 13],
                         [1, 2, 13],
                         [2, 13, 14],
                         [2, 3, 14],
                         [3, 14, 15], #end of left side, start of top
                         [0, 7, 12],
                         [7, 12, 19], #end of top, start of bottom
                         [3, 15, 16],
                         [3, 4, 16], #end of shape
                         ], dtype='i8')

utils.set_polydata_vertices(my_polydata, my_vertices)
utils.set_polydata_triangles(my_polydata, my_triangles)

file_name = "my_rhombicube.vtk"
save_polydata(my_polydata, file_name)
print("Surface saved in " + file_name)

rhombicube_polydata = load_polydata(file_name)

rhombicube_vertices = utils.get_polydata_vertices(rhombicube_polydata)
colors = rhombicube_vertices * 255
utils.set_polydata_colors(rhombicube_polydata, colors)

print("new surface colors")
print(utils.get_polydata_colors(rhombicube_polydata))

# get vtkActor
rhombicube_actor = utils.get_actor_from_polydata(rhombicube_polydata)
rhombicube_actor.GetProperty().BackfaceCullingOff() #gets rid of the winding order issue (look at later and other algorithms that get rid of winding order)

# Create a scene
scene = window.Scene()
scene.add(rhombicube_actor)
scene.set_camera(position=(0, 0, 7), focal_point=(0, 0, 0))
scene.zoom(0)

# display
window.show(scene, size=(1000, 1000), reset_camera=False) #this allows the picture to be moved around
window.record(scene, out_path='rhombicu.png', size=(600, 600))