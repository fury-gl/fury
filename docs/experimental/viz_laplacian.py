import numpy as np

from fury.actor import cube, sphere
from fury.actors.effect_manager import EffectManager
from fury.shaders import shader_apply_effects, shader_custom_uniforms
from fury.window import (Scene, ShowManager, record)

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
scene.set_camera(position=(0, 0, -10),
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

cube_actor = cube(np.array([[0.0, 0.0, -3.0]]), colors = (1.0, 0.5, 0.0))
sphere_actor = sphere(np.array([[0.0, 0.0, 3.0]]), (0.0, 1.0, 1.0), radii = 2.0)

effects = EffectManager(manager)

lapl_actor = effects.gaussian_blur(cube_actor, 4.0, 1.0)
lapl_sphere = effects.grayscale(sphere_actor, 4.0, 1.0)

manager.scene.add(lapl_sphere)
manager.scene.add(lapl_actor)

interactive = True

if interactive:
    manager.start()

# effects.remove_effect(lapl_actor)

# record(scene, out_path = "kde_points.png", size = (800, 800))