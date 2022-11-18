"""
============================
Morphing Animation in a glTF
============================
In this tutorial, we will show how to use morphing in a glTF model in FURY.
"""

from fury import window
from fury.gltf import glTF
from fury.data import fetch_gltf, read_viz_gltf

##############################################################################
# Retrieving the model with morphing in it (look at Khronoos samples).
# We're choosing the `MorphStressTest` model here.

fetch_gltf('MorphStressTest', 'glTF')
filename = read_viz_gltf('MorphStressTest')

##############################################################################
# Initializing the glTF object, You can additionaly set `apply_normals=True`.
# Note: Normals might not work as intended with morphing.

gltf_obj = glTF(filename, apply_normals=False)

##############################################################################
# Get the morph timeline using `morph_timeline` method, Choose the animation
# name you want to visualize.
# Note: If there's no name for animation, It's stored as `anim_0`, `anim_1` etc

timeline = gltf_obj.morph_timeline()['TheWave']

##############################################################################
# Call the `update_morph` method once, This moves initialise the morphing at
# timestamp 0.0 seconds and ensures that camera fits all the actors perfectly.

gltf_obj.update_morph(timeline)

##############################################################################
# Create a scene, and show manager.
# Initialize the show manager and add timeline to the scene (No need to add
# actors to the scene seperately).

scene = window.Scene()
showm = window.ShowManager(scene, size=(900, 768), reset_camera=True,
                           order_transparent=True)

showm.initialize()
scene.add(timeline)

##############################################################################
# define a timer_callback.
# Use the `update_morph` method again, It updates the timeline and applies
# morphing).


def timer_callback(_obj, _event):
    gltf_obj.update_morph(timeline)
    showm.render()


##############################################################################
# Optional: `timeline.play()` auto plays the animations.

timeline.play()

showm.add_timer_callback(True, 20, timer_callback)
scene.reset_camera()

interactive = False

if interactive:
    showm.start()

window.record(scene, out_path='viz_morphing.png',
              size=(900, 768))
