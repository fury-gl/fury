"""
======================================================================
Fury Kernel Density Estimation rendering Actor
======================================================================
This example shows how to use the KDE effect. This is a feature in Fury
that uses a post-processing stage to render kernel density estimations of a 
given set of points in real-time to the screen. For better understanding 
on KDEs, check this `Wikipedia page <https://en.wikipedia.org/wiki/Kernel_density_estimation>`_ 
about it.

For this example, you will only need the modules below:
"""
import numpy as np

from fury.effects import EffectManager, KDE
from fury.window import Scene, ShowManager, record

#####################################################################################
# This function below will help us to relocate the points for a better visualization,
# but it is not required.

def normalize(array : np.array, min : float = 0.0, max : float = 1.0, axis : int = 0):
    """Convert an array to a given desired range.

    Parameters
    ----------
    array : np.ndarray
        Array to be normalized.
    min : float
        Bottom value of the interval of normalization. If no value is given, it is passed as 0.0.
    max : float
        Upper value of the interval of normalization. If no value is given, it is passed as 1.0.

    Returns
    -------
    array : np.array
        Array converted to the given desired range.
    """
    if np.max(array) != np.min(array):
        return ((array - np.min(array))/(np.max(array) - np.min(array)))*(max - min) + min
    else:
        raise ValueError(
            "Can't normalize an array which maximum and minimum value are the same.")

##################################################################
# First, we need to setup the screen we will render the points to.

size = (1200, 1000)

scene = Scene()
scene.set_camera(position=(-24, 20, -40),
                 focal_point=(0.0,
                              0.0,
                              0.0),
                 view_up=(0.0, 0.0, 1.0))

manager = ShowManager(
    scene,
    "KDE Render",
    size)


####################################################################
# ``numpy.random.rand`` will be used to generate random points, which
# will be then relocated with the function we declared below to the
# range of ``[-5.0, 5.0]``, so they get more space between them. In case
# offsetted points are wanted, it can be done just as below.

n_points = 1000
points = np.random.rand(n_points, 3)
points = normalize(points, -5, 5)
offset = np.array([0.0, 0.0, 0.0])
points = points + np.tile(offset, points.shape[0]).reshape(points.shape)

###################################################################
# For this KDE render, we will use a set of random bandwidths,
# generated with ``numpy.random.rand`` as well, which are also
# remapped to the range of ``[0.05, 0.2]``.

bandwidths = normalize(np.random.rand(n_points, 1), 0.05, 0.2)

###################################################################
# Now, for the KDE render, a special class is needed, the
# ``EffectManager``. This class is needed to manage the post-processing
# aspect of this kind of render, as it will need to first be
# rendered to an offscreen buffer, retrieved and then processed
# by the final actor that will render it to the screen, but don't
# worry, none of this will need to be setup by you! Just call the
# ``EffectManager`` like below, passing the manager to it:

effects_m = EffectManager(manager)

###################################################################
# After having the ``effects_m`` setup, we can add effects to it.
# Here, we will use the ``KDE`` effect, that comes from ``fury.effects.effects``
# module. When we call it, we pass the points, bandwidth, and other optional options
# if wished, like the kernel to be used or the colormap desired. The available kernels
# can be found inside KDE's description. The colormaps are by default taken from *matplotlib*, 
# but a custom one can be passed. After calling it, it returns the KDE class, that will
# then need to be added to the ``EffectManager`` with the ``add()`` method.

kde_effect = KDE(points, bandwidths, kernel="exponential", colormap="inferno")
effects_m.add(kde_effect)

####################################################################
# Having that setup, just start the rendering process to see the results, and we are done here!

interactive = False

if interactive:
    manager.start()

record(scene, out_path="kde_points.png", size=(800, 800))
