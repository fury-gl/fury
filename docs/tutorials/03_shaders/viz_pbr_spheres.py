from fury import actor, material, window
from fury.utils import (get_polydata_normals, normalize_v3,
                        set_polydata_tangents)
import numpy as np

scene = window.Scene()
scene.background((.9, .9, .9))

material_params = [
    [[1, 1, 0], {'metallic': 0, 'roughness': 0}],
    [(0, 0, 1), {'roughness': 0}],
    [(1, 0, 1), {'anisotropy': 0, 'metallic': .25, 'roughness': .5}],
    [(1, 0, 1), {'anisotropy_rotation': 0, 'anisotropy': 1, 'metallic': .25,
                 'roughness': .5}]
]

for i in range(4):
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

labels = ['Metallic', 'Roughness', 'Anisotropy', 'Anisotropy Rotation']

for i in range(4):
    pos = [-40, -5 * i, 0]
    label = actor.label(labels[i], pos=pos, scale=(.8, .8, .8),
                        color=(0, 0, 0))
    scene.add(label)

for j in range(11):
    pos = [-26 + 5 * j, 5, 0]
    label = actor.label(str(np.round(j * 0.1, decimals=1)), pos=pos,
                        scale=(.8, .8, .8), color=(0, 0, 0))
    scene.add(label)

window.show(scene)
