import numpy as np

from fury.actors.effect_manager import EffectManager
from fury.window import Scene, ShowManager, record

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


width, height = (1200, 1000)

scene = Scene()
scene.set_camera(position=(-6, 5, -10),
                 focal_point=(0.0,
                              0.0,
                              0.0),
                 view_up=(0.0, 0.0, 0.0))

manager = ShowManager(
    scene,
    "demo",
    (width,
     height))

manager.initialize()


n_points = 1000
points = np.random.rand(n_points, 3)
points = normalize(points, -5, 5)
sigmas = normalize(np.random.rand(n_points, 1), 0.1, 0.6)
offset = np.array([0.0, 0.0, 0.0])
points = points + np.tile(offset, points.shape[0]).reshape(points.shape)

effects = EffectManager(manager)

kde_actor = effects.kde(points, sigmas, kernel = "exponential", colormap = "inferno")


manager.scene.add(kde_actor)
# effects.remove_effect(kde_actor)

interactive = True

if interactive:
    manager.start()
    
record(scene, out_path = "kde_points.png", size = (800, 800))
