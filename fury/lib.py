import vtkmodules.vtkCommonCore as ccvtk
import vtkmodules.vtkRenderingCore as rcvtk
import vtkmodules.vtkFiltersHybrid as fhvtk
import vtkmodules.vtkInteractionStyle as isvtk
from vtkmodules.util import numpy_support, colors


VTK_9_PLUS = ccvtk.vtkVersion.GetVTKMajorVersion() >= 9

##############################################################
#  vtkCommonCore Module

Command = ccvtk.vtkCommand


##############################################################
#  vtkRenderingCore Module
Renderer = rcvtk.vtkRenderer
Volume = rcvtk.vtkVolume
Actor2D = rcvtk.vtkActor2D
RenderWindow = rcvtk.vtkRenderWindow
RenderWindowInteractor = rcvtk.vtkRenderWindowInteractor
InteractorEventRecorder = rcvtk.vtkInteractorEventRecorder
WindowToImageFilter = rcvtk.vtkWindowToImageFilter

##############################################################
#  vtkInteractionStyle Module

InteractorStyleImage = isvtk.vtkInteractorStyleImage
InteractorStyleTrackballCamera = isvtk.vtkInteractorStyleTrackballCamera

##############################################################
#  vtkFiltersHybrid Module
RenderLargeImage = fhvtk.vtkRenderLargeImage
