# from fury import actor, window
from fury import window
import vtk


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
    return act


lines = vtk.vtkLineSource()
lines.SetResolution(11)
start_pos = [0, 0, 0]
end_pos = [1, 1, 0]
lines.SetPoint1(start_pos)
lines.SetPoint2(end_pos)

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(lines.GetOutputPort())

line_actor = vtk.vtkActor()
line_actor.SetMapper(mapper)
line_actor.GetProperty().SetColor([1, 0, 0])

StippledLine(line_actor, 0xAAAA, 2)


scene = window.Scene()
scene.add(line_actor)
window.show(scene)
