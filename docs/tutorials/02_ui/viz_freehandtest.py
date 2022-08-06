# Pure VTK Code
# import vtk

# imageSource = vtk.vtkImageCanvasSource2D()

# imageSource.SetScalarTypeToUnsignedChar()
# imageSource.SetNumberOfScalarComponents(3)
# imageSource.SetExtent(0, 20, 0, 50, 0, 0)
# imageSource.SetDrawColor(0, 0, 0)
# imageSource.FillBox(0, 20, 0, 50)
# imageSource.SetDrawColor(255, 0, 0)
# imageSource.FillBox(0, 10, 0, 30)
# imageSource.Update()

# actor = vtk.vtkImageActor()
# actor.GetMapper().SetInputConnection(imageSource.GetOutputPort())
# actor.VisibilityOn()
# actor.AddPosition(10, 10, -13)
# actor.InterpolateOff()

# ip = vtk.vtkImageProperty()
# ip.SetColorWindow(2000)
# ip.SetColorLevel(1000)
# ip.SetAmbient(0.0)
# ip.SetDiffuse(1.0)
# ip.SetOpacity(1.0)
# ip.SetInterpolationTypeToLinear()

# actor.SetProperty(ip)

# renderer = vtk.vtkRenderer()

# renderer.AddActor(actor)
# renderer.ResetCamera()


# renderWindow = vtk.vtkRenderWindow()
# renderWindow.AddRenderer(renderer)

# interactor = vtk.vtkRenderWindowInteractor()
# interactor.SetRenderWindow(renderWindow)

# style = vtk.vtkInteractorStyleImage()
# interactor.SetInteractorStyle(style)

# tracer = vtk.vtkImageTracerWidget()

# tracer.SetInteractor(interactor)
# tracer.SetViewProp(actor)
# tracer.AutoCloseOn()

# renderWindow.Render()
# tracer.On()

# vtk.vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()
# vtk.vtkMapper.SetResolveCoincidentTopologyPolygonOffsetParameters(10, 10)


# interactor.Start()


#######################################################################################
# FURY Manager Code

from fury.data import read_viz_icons
from fury import ui, window

import vtk

imageSource = vtk.vtkImageCanvasSource2D()

imageSource.SetScalarTypeToUnsignedChar()
imageSource.SetNumberOfScalarComponents(3)
imageSource.SetExtent(0, 20, 0, 50, 0, 0)
imageSource.SetDrawColor(0, 0, 0)
imageSource.FillBox(0, 20, 0, 50)
imageSource.SetDrawColor(255, 0, 0)
imageSource.FillBox(0, 10, 0, 30)
imageSource.Update()

actor = vtk.vtkImageActor()
actor.GetMapper().SetInputConnection(imageSource.GetOutputPort())
actor.VisibilityOn()
actor.AddPosition(10, 10, -13)
actor.InterpolateOff()

ip = vtk.vtkImageProperty()
ip.SetColorWindow(2000)
ip.SetColorLevel(1000)
ip.SetAmbient(0.0)
ip.SetDiffuse(1.0)
ip.SetOpacity(1.0)
ip.SetInterpolationTypeToLinear()

actor.SetProperty(ip)

# Using Fury Actor
# img_container = ui.ImageContainer2D(img_path=read_viz_icons(fname='pencil.png'))
# actor = img_container.actor

sm = window.ShowManager()
sm.scene.add(actor)

tracer = vtk.vtkImageTracerWidget()

tracer.SetInteractor(sm.iren)
tracer.SetViewProp(actor)
tracer.AutoCloseOn()

tracer.On()


sm.start()
