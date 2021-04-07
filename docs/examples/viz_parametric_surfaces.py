"""
Parametric surfaces
================================

This is a simple demonstration of how users can generate and render parametric
surfaces by using the parametric_surface actor.

Parametric surfaces visualized in this tutorial -
Möbius strip
Klein bottle
Roman surface
Boy's surface
Bohemian Dome
Dini's surface
Plücker's conoid
"""

###############################################################################
# Importing necessary modules

from fury import window, actor, ui
import numpy as np

###############################################################################
# List to store names of the parametric surfaces which will be rendered


s_names = ["mobius_strip", "kleins_bottle", "roman_surface", "boys_surface",
           "bohemian_dome", "dinis_surface", "pluckers_conoid"]
centers = np.array([[-12, 0, 0], [-8, 0, 0], [-4, 0, 0], [0, 0, 0], [4, 0, 0],
                    [8, 0, 0], [12, 0, 0]])


###############################################################################
# Creating a scene object and configuring the camera's position


scene = window.Scene()
scene.background((1, 1, 1))
scene.zoom(5.5)
scene.set_camera(position=(-10, 10, -150), focal_point=(0.0, 0.0, 0.0),
                 view_up=(0.0, 0.0, 0.0))

showm = window.ShowManager(scene,
                           size=(600, 600), reset_camera=True,
                           order_transparent=True)


###############################################################################
# Adding the actors to the scene

for i, name in enumerate(s_names):
    scale = 1
    # Increase the scale for Roman surface as it's relatively smaller than the
    # other surfaces
    if name == 'roman_surface':
        scale = 3
    scene.add(actor.parametric_surface(np.array([centers[i]]), name=name,
              colors=np.random.rand(3), scales=scale))


###############################################################################
# Textbox for title of the demo

tb = ui.TextBlock2D(bold=True, font_size=20, position=(200, 470),
                    color=(0, 0, 0))
tb.message = "Some Parametric Objects"
scene.add(tb)

interactive = False
if interactive:
    window.show(scene, size=(600, 600))
window.record(showm.scene, size=(600, 600),
              out_path="viz_parametric_surfaces.png")
