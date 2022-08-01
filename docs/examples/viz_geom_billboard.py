import numpy as np
from fury import window, actor
from fury.shaders import shader_to_actor, import_fury_shader

centers = np.random.random([3000, 3]) * 50
colors = np.random.random([3000, 3])
scales = np.random.random(3000)

scene = window.Scene()
showm = window.ShowManager(scene, size=(1000, 768))

geom_squares = actor.billboard(centers, colors=colors, scales=scales,
                               using_gs=True)

shader_to_actor(geom_squares, 'fragment',
                impl_code=import_fury_shader('gs_billboard_sphere_impl.frag'))

scene.add(geom_squares)

showm.start()
