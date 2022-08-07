"""
=======================
Visualizing a glTF file
=======================
In this tutorial, we will show how to display a glTF file in a scene.
"""

from fury import window
from fury.gltf import glTF
from fury.data import fetch_gltf, read_viz_gltf

##############################################################################
# Create a scene.

scene = window.Scene()

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)
showm.initialize()


##############################################################################
# Retrieving the gltf model.
fetch_gltf('CesiumMilkTruck', 'glTF')
filename = read_viz_gltf('CesiumMilkTruck')

##############################################################################
# Initialize the glTF object and get actors using `actors` method.
# Note: You can always manually create actor from polydata, and apply texture
# or materials manually afterwards.
# Experimental: For smooth mesh/actor you can set `apply_normals=True`.

gltf_obj = glTF(filename)
timeline = gltf_obj.get_main_timeline()

##############################################################################
# Add all the actor from list of actors to the scene.

scene.add(timeline)

##############################################################################
# Applying camera


def timer_callback(_obj, _event):
    timeline.update_animation()
    showm.render()


showm.add_timer_callback(True, 10, timer_callback)

showm.start()
