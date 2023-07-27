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

offset = 1.0
cube_actor = cube(np.array([[0.0, 0.0, -1.0 + offset]]), colors = (1.0, 0.5, 0.0))
sphere_actor = sphere(np.array([[0.0, 0.0, 1.0 + offset], [1.0, 0.5, 1.0 + offset]]), (0.0, 1.0, 1.0), radii = 0.5)

effects = EffectManager(manager)
gauss_cube = effects.gaussian_blur(cube_actor, 1.0)
gray_sphere = effects.grayscale(sphere_actor, 1.0)

manager.scene.add(gray_sphere)
manager.scene.add(gauss_cube)

# effects.remove_effect(gauss_cube)

interactive = False

if interactive:
    manager.start()

record(scene, out_path = "post_process.png", size = (800, 800))