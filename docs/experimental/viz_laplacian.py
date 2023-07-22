import numpy as np

from fury.actor import cube
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


width, height = (800, 800)

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

cube_actor = cube(np.array([[0.0, 0.0, 0.0]]), colors = (1.0, 0.5, 0.0))

effects = EffectManager(manager)

lapl_actor = effects.laplacian(np.array([[0.0, 0.0, 0.0]]), cube_actor, 4.0, 1.0)

lapl2 = effects.laplacian(np.array([[0.0, 0.0, 0.0]]), lapl_actor, 4.0, 1.0)

# manager.scene.add(cu)
manager.scene.add(lapl2)

interactive = True

if interactive:
    manager.start()

effects.remove_effect(lapl_actor)
effects.remove_effect(lapl2)

# record(scene, out_path = "kde_points.png", size = (800, 800))

if interactive:
    manager.start()