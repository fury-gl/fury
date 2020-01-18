"""
===============
Spiky Sphere
===============




"""

import numpy as np
from fury import window, actor, ui, utils, primitive
vertices, triangles = primitive.prim_sphere()
normals = utils.normals_from_v_f(vertices, triangles) 

centers_sphere = np.array([[0, 2, 0.]]) 
colors_sphere = np.array([[1, .3, 0.]])
radii = np.random.rand(100) + 0.5
scene = window.Scene()
sphere_actor = actor.sphere(centers=centers_sphere,
                            colors=colors_sphere,
                            radii=radii)
cone_actor = actor.arrow(centers=vertices,
                         directions=normals, colors=(1, 0, 0), heights=0.2,
                         resolution=10, vertices=None, faces=None)

primitive_colors = np.zeros(vertices.shape)
primitive_colors[:, 2] = 180
primitive_actor = utils.get_actor_from_primitive(
    vertices=vertices, triangles=triangles, colors=primitive_colors,
    normals=normals, backface_culling=True)
#scene.add(sphere_actor)
scene.add(cone_actor)
scene.add(primitive_actor)
showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)


showm.initialize()
showm.start()

