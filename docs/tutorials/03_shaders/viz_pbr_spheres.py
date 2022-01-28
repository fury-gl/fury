"""
===============================================================================
Physically-Based Rendering (PBR) on spheres
===============================================================================

PBR engines aim to simulate properties of light when it interacts with objects
in the scene in a physically plausible way. The interaction of light with an
object depends on the material the object is made of. In computer graphics,
materials are usually divided in 2 main categories based on their conductive
properties: dielectrics and metals.

This tutorial, illustrates how to model some material properties in FURY by
using the PBR material.

Let's start by importing the necessary modules:
"""

from fury import actor, material, window
from fury.utils import (normals_from_actor, tangents_to_actor,
                        tangents_from_direction_of_anisotropy)
import numpy as np

"""
Now set up a new scene.
"""

scene = window.Scene()
scene.background((.9, .9, .9))

"""
Let's define the parameters we are going to showcase in this tutorial.
These subset of parameters have their values constrained in the 0 to 1 range.
"""

material_params = [
    [[1, 1, 0], {'metallic': 0, 'roughness': 0}],
    [(0, 0, 1), {'roughness': 0}],
    [(1, 0, 1), {'anisotropy': 0, 'metallic': .25, 'roughness': .5}],
    [(1, 0, 1), {'anisotropy_rotation': 0, 'anisotropy': 1, 'metallic': .25,
                 'roughness': .5}],
    [(0, 1, 1), {'coat_strength': 0, 'roughness': 0}],
    [(0, 1, 1), {'coat_roughness': 0, 'coat_strength': 1, 'roughness': 0}]
]

"""
Now we can start to add our actors to the scene and see how different values of
the parameters produce interesting effects. For the purpose of this tutorial,
we will see the effect of 11 different values of each parameter.
"""

num_values = 11

for i, mp in enumerate(material_params):
    color = mp[0]
    params = mp[1]
    center = [[0, -5 * i, 0]]
    for j in range(num_values):
        center[0][0] = -25 + 5 * j
        sphere = actor.sphere(center, color, radii=2, theta=32, phi=32)
        normals = normals_from_actor(sphere)
        tangents = tangents_from_direction_of_anisotropy(normals, (0, 1, .5))
        tangents_to_actor(sphere, tangents)
        keys = list(params)
        params[keys[0]] = np.round(0.1 * j, decimals=1)
        material.manifest_pbr(sphere, **params)
        scene.add(sphere)

"""
For interpretability purposes we will add some labels to guide us through our
visualization.
"""

labels = ['Metallic', 'Roughness', 'Anisotropy', 'Anisotropy Rotation',
          'Coat Strength', 'Coat Roughness']

for i, l in enumerate(labels):
    pos = [-40, -5 * i, 0]
    label = actor.vector_text(l, pos=pos, scale=(.8, .8, .8), color=(0, 0, 0))
    scene.add(label)

for j in range(num_values):
    pos = [-26 + 5 * j, 3, 0]
    label = actor.vector_text(str(np.round(j * 0.1, decimals=1)), pos=pos,
                              scale=(.8, .8, .8), color=(0, 0, 0))
    scene.add(label)

"""
Some parameters of this material have their values constrained to be between 1
and 2.3. These parameters are the Base Index of Refraction (IOR) and the Clear
coat Index of Refraction (IOR). Therefore, we will interpolate some values
within this range and see how they affect the rendering.
"""

iors = np.round(np.linspace(1, 2.3, num=num_values), decimals=2)

ior_params = [
    [(0, 1, 1), {'base_ior': iors[0], 'roughness': 0}],
    [(0, 1, 1), {'coat_ior': iors[0], 'coat_roughness': .1, 'coat_strength': 1,
                 'roughness': 0}]
]

for i, iorp in enumerate(ior_params):
    color = iorp[0]
    params = iorp[1]
    center = [[0, -35 - (5 * i), 0]]
    for j in range(num_values):
        center[0][0] = -25 + 5 * j
        sphere = actor.sphere(center, color, radii=2, theta=32, phi=32)
        keys = list(params)
        params[keys[0]] = iors[j]
        material.manifest_pbr(sphere, **params)
        scene.add(sphere)

"""
Let's add the respective labels to the scene.
"""

labels = ['Base IoR', 'Coat IoR']

for i, l in enumerate(labels):
    pos = [-40, -35 - (5 * i), 0]
    label = actor.vector_text(l, pos=pos, scale=(.8, .8, .8), color=(0, 0, 0))
    scene.add(label)

for j in range(num_values):
    pos = [-26 + 5 * j, -32, 0]
    label = actor.vector_text('{:.02f}'.format(iors[j]), pos=pos,
                              scale=(.8, .8, .8), color=(0, 0, 0))
    scene.add(label)

"""
Finally, let's visualize our tutorial.
"""

interactive = False
if interactive:
    window.show(scene)

window.record(scene, size=(600, 600), out_path="viz_pbr_spheres.png")
