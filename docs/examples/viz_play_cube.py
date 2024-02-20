"""
=======================================================
Play a video in the 3D world
=======================================================

The goal of this demo is to show how to visualize a video
on a cube by updating a texture.
"""

import cv2
import vtk

import numpy as np 

from fury import actor, window
from fury.lib import numpy_support

def texture_on_cube(image):
    """
    Map an RGB texture on a cube.

    Parameters:
    -----------
    image : ndarray
        Input 2D RGB or RGBA array. Dtype should be uint8.
    
    Returns:
    --------
    actor : Actor
    """

    grid = vtk.vtkImageData()
    grid.SetDimensions(image.shape[1], image.shape[0], 1)
    vtkarr = numpy_support.numpy_to_vtk(np.flip(image.swapaxes(0,1), axis=1).reshape((-1, 3), order='F'))
    vtkarr.SetName('Image')
    grid.GetPointData().AddArray(vtkarr)
    grid.GetPointData().SetActiveScalars('Image')

    vtex = vtk.vtkTexture()
    vtex.SetInputDataObject(grid)
    vtex.Update()

    cubeSource = vtk.vtkCubeSource()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(cubeSource.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetTexture(vtex)

    return actor

# timer_callback is called by window.showManager
def timer_callback(caller, timer_event):
    _, image = cam.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    actor.texture_update(cube, image)
    showmanager.render()

# openCV video capture and conversion to RGB
cam = cv2.VideoCapture('http://commondatastorage.googleapis.com/'
    + 'gtv-videos-bucket/sample/BigBuckBunny.mp4')
_, image = cam.read()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Scene creation
scene = window.Scene()

# actor for fury
cube = texture_on_cube(image)

# working with window.ShowManager to setup timer_callbacks
scene.add(cube)
showmanager = window.ShowManager(scene, size=(600, 600), reset_camera=False)
showmanager.add_timer_callback(True, int(1/60), timer_callback)
showmanager.start()