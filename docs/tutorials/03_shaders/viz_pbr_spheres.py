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
from fury.utils import (get_polydata_normals, normalize_v3,
                        set_polydata_tangents)
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
the parameters produce interesting effects.
"""

for i in range(6):
    center = [[0, -5 * i, 0]]
    for j in range(11):
        center[0][0] = -25 + 5 * j
        sphere = actor.sphere(center, material_params[i][0], radii=2, theta=32,
                              phi=32)
        polydata = sphere.GetMapper().GetInput()
        normals = get_polydata_normals(polydata)
        tangents = np.cross(normals, np.array([0, 1, .5]))
        binormals = normalize_v3(np.cross(normals, tangents))
        tangents = normalize_v3(np.cross(normals, binormals))
        set_polydata_tangents(polydata, tangents)
        keys = list(material_params[i][1])
        material_params[i][1][keys[0]] = np.round(0.1 * j, decimals=1)
        material.manifest_pbr(sphere, **material_params[i][1])
        scene.add(sphere)

"""
For interpretability purposes we will add some labels to guide us through our 
visualization.
"""

labels = ['Metallic', 'Roughness', 'Anisotropy', 'Anisotropy Rotation',
          'Coat Strength', 'Coat Roughness']

for i in range(6):
    pos = [-40, -5 * i, 0]
    label = actor.vector_text(labels[i], pos=pos, scale=(.8, .8, .8),
                              color=(0, 0, 0))
    scene.add(label)

for j in range(11):
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

iors = np.round(np.linspace(1, 2.3, num=11), decimals=2)

ior_params = [
    [(0, 1, 1), {'base_ior': iors[0], 'roughness': 0}],
    [(0, 1, 1), {'coat_ior': iors[0], 'coat_roughness': .1, 'coat_strength': 1,
                 'roughness': 0}]
]

for i in range(2):
    center = [[0, -35 - (5 * i), 0]]
    for j in range(11):
        center[0][0] = -25 + 5 * j
        sphere = actor.sphere(center, ior_params[i][0], radii=2, theta=32,
                              phi=32)
        keys = list(ior_params[i][1])
        ior_params[i][1][keys[0]] = iors[j]
        material.manifest_pbr(sphere, **ior_params[i][1])
        scene.add(sphere)

"""
Let's add the respective labels to the scene.
"""

labels = ['Base IoR', 'Coat IoR']

for i in range(2):
    pos = [-40, -35 - (5 * i), 0]
    label = actor.vector_text(labels[i], pos=pos, scale=(.8, .8, .8),
                              color=(0, 0, 0))
    scene.add(label)

for j in range(11):
    pos = [-26 + 5 * j, -32, 0]
    label = actor.vector_text('{:.02f}'.format(iors[j]), pos=pos,
                              scale=(.8, .8, .8), color=(0, 0, 0))
    scene.add(label)

"""
Finally, let's visualize our demo.
"""

interactive = False
if interactive:
    window.show(scene)

window.record(scene, size=(600, 600), out_path="viz_pbr_spheres.png")