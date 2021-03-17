# from fury import actor, window
from fury import window
import numpy as np
import vtk
from fury.utils import lines_to_vtk_polydata, set_input


def StippledLine(act, lineStipplePattern,
                 lineStippleRepeat):
    image = vtk.vtkImageData()
    texture = vtk.vtkTexture()

    dimension = 16 * lineStippleRepeat

    image.SetDimensions(dimension, 1, 1)
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)
    image.SetExtent(0, dimension - 1, 0, 0, 0, 0)
    on = 255
    off = 0
    i_dim = 0
    while i_dim < dimension:
        for i in range(0, 16):
            mask = (1 << i)
            bit = (lineStipplePattern & mask) >> i
            value = bit
            if value == 0:
                for j in range(0, lineStippleRepeat):
                    image.SetScalarComponentFromFloat(i_dim, 0, 0, 0, on)
                    image.SetScalarComponentFromFloat(i_dim, 0, 0, 1, on)
                    image.SetScalarComponentFromFloat(i_dim, 0, 0, 2, on)
                    image.SetScalarComponentFromFloat(i_dim, 0, 0, 3, off)
                    i_dim += 1
            else:
                for j in range(0, lineStippleRepeat):
                    image.SetScalarComponentFromFloat(i_dim, 0, 0, 0, on)
                    image.SetScalarComponentFromFloat(i_dim, 0, 0, 1, on)
                    image.SetScalarComponentFromFloat(i_dim, 0, 0, 2, on)
                    image.SetScalarComponentFromFloat(i_dim, 0, 0, 3, on)
                    i_dim += 1

    texture.SetInputData(image)
    texture.InterpolateOff()
    texture.RepeatOn()

    act.SetTexture(texture)
    # print(act.IsA('vtkActor'))
    # print(act.GetLODMappers())
    return act


lines = [np.random.rand(2, 3), np.random.rand(2, 3)]
colors = None
linewidth = 1
lod_points = 10 ** 4
lod_points_size = 3

poly_data, color_is_scalar = lines_to_vtk_polydata(lines, colors)
next_input = poly_data
poly_mapper = set_input(vtk.vtkPolyDataMapper(), next_input)

act = vtk.vtkLODActor()
act.SetNumberOfCloudPoints(lod_points)
act.GetProperty().SetPointSize(lod_points_size*2)

act.SetMapper(poly_mapper)
act.GetProperty().SetLineWidth(linewidth)

StippledLine(act, 0xAAAA, 2)


scene = window.Scene()
scene.add(act)
# window.show(scene)

