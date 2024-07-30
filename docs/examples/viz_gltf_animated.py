"""
=======================
Visualizing a glTF file
=======================
In this tutorial, we will show how to display a simple animated glTF in a
scene.
"""

import fury

##############################################################################
# Create a scene.

scene = fury.window.Scene()

showm = fury.window.ShowManager(
    scene, size=(900, 768), reset_camera=False, order_transparent=True
)
showm.initialize()


##############################################################################
# Retrieving the gltf model.
fury.data.fetch_gltf("InterpolationTest", "glTF")
filename = fury.data.read_viz_gltf("InterpolationTest")

##############################################################################
# Initialize the glTF object and get actors using `actors` method.
# Get the main_timeline (which contains multiple Timeline objects).

gltf_obj = fury.gltf.glTF(filename)
timeline = gltf_obj.main_animation()

##############################################################################
# Add the timeline to the scene (No need to add actors separately).

scene.add(timeline)

##############################################################################
# define a timer_callback that updates the timeline.

interactive = False


def timer_callback(_obj, _event):
    timeline.update_animation()
    showm.render()


showm.add_timer_callback(True, 10, timer_callback)

if interactive:
    showm.start()

fury.window.record(scene, out_path="viz_gltf_animated.png", size=(900, 768))
