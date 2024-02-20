"""
=======================================================
Play a video in the 3D world
=======================================================

The goal of this demo is to show how to visualize a video
on a cube by updating a texture.
"""

import cv2
import vtk  # only for vtkCubeSource which needs to be added to lib.py

import numpy as np

from fury import actor, window
from fury.lib import (
    numpy_support,
    ImageData,
    Texture,
    # lib.py needs to have CubeSource,
    PolyDataMapper,
    Actor
)


def texture_on_cube(image):
    """
    Map an RGB texture on a cube.

    Parameters:
    -----------
    image : ndarray
        Input 2D RGB array. Dtype should be uint8.

    Returns:
    --------
    actor : Actor
    """

    grid = ImageData()
    grid.SetDimensions(image.shape[1], image.shape[0], 1)
    # we need a numpy array -> vtkTexture function in numpy_support
    vtkarr = numpy_support.numpy_to_vtk(np.flip(image.swapaxes(0, 1), axis=1).reshape((-1, 3), order='F'))
    vtkarr.SetName('Image')
    grid.GetPointData().AddArray(vtkarr)
    grid.GetPointData().SetActiveScalars('Image')

    vtex = Texture()
    vtex.SetInputDataObject(grid)
    vtex.Update()

    cubeSource = vtk.vtkCubeSource()

    mapper = PolyDataMapper()
    mapper.SetInputConnection(cubeSource.GetOutputPort())

    actor = Actor()
    actor.SetMapper(mapper)
    actor.SetTexture(vtex)

    return actor


# timer_callback is called by window.showManager
def timer_callback(caller, timer_event):
    _, image = cam.read()

    if image is None:
        showmanager.exit()
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        actor.texture_update(cube, image)
        showmanager.render()


# openCV video capture and conversion to RGB
cam = cv2.VideoCapture('http://commondatastorage.googleapis.com/'
                        + 'gtv-videos-bucket/sample/BigBuckBunny.mp4')
fps = int(cam.get(cv2.CAP_PROP_FPS))

_, image = cam.read()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Scene creation
scene = window.Scene()

# actor for fury
cube = texture_on_cube(image)

# working with window.ShowManager to setup timer_callbacks
scene.add(cube)
showmanager = window.ShowManager(scene, size=(600, 600), reset_camera=False)
showmanager.add_timer_callback(True, int(1000/fps), timer_callback)
showmanager.start()
