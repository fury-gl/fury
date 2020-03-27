from fury import actor, window, io
import numpy as np
import numpy.testing as npt
import os

from tempfile import TemporaryDirectory as InTemporaryDirectory


xyzr = np.array([[0, 0, 0, 10], [100, 0, 0, 25], [200, 0, 0, 50]])
colors = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1., 1]])
sphere_actor = actor.sphere(centers=xyzr[:, :3], colors=colors[:],
                            radii=xyzr[:, 3])
scene = window.Scene()
scene.add(sphere_actor)


with InTemporaryDirectory():
    window.record(scene, out_path='fury_1.png', size=(1000, 1000),
                  magnification=5)
    npt.assert_equal(os.path.isfile('fury_1.png'), True)
    arr = io.load_image('fury.png')
    print(arr.shape)