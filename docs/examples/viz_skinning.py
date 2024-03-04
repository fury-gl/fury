"""
=================================
Skeletal Animation in a glTF file
=================================
In this tutorial, we will show how to use skeletal animations (skinning) in a
glTF model in FURY.
"""

from fury import window
from fury.data import fetch_gltf, read_viz_gltf
from fury.gltf import glTF

##############################################################################
# Retrieving the model with skeletal animations.
# We're choosing the `RiggedFigure` model here.

fetch_gltf('RiggedFigure', 'glTF')
filename = read_viz_gltf('RiggedFigure')

##############################################################################
# Initializing the glTF object, You can additionally set `apply_normals=True`.
# Note: Normals might not work well as intended with skinning animations.

gltf_obj = glTF(filename, apply_normals=False)

##############################################################################
# Get the skinning timeline using `skin_timeline` method, Choose the animation
# name you want to visualize.
# Note: If there's no name for animation, It's stored as `anim_0`, `anim_1` etc

animation = gltf_obj.skin_animation()['anim_0']

# After we get the timeline object, We want to initialise the skinning process.
# You can set `bones=true` to visualize each bone transformation. Additionally,
# you can set `length` of bones in the `initialise_skin` method.
# Note: Make sure to call this method before you initialize ShowManager, else
# bones won't be added to the scene.

gltf_obj.initialize_skin(animation, bones=False)

##############################################################################
# Create a scene, and show manager.
# Initialize the show manager and add timeline to the scene (No need to add
# actors to the scene separately).

scene = window.Scene()
showm = window.ShowManager(
    scene, size=(900, 768), reset_camera=True, order_transparent=True
)
showm.initialize()
scene.add(animation)

##############################################################################
# define a timer_callback.
# Use the `update_skin` method, It updates the timeline and applies skinning to
# actors (and bones).


def timer_callback(_obj, _event):
    gltf_obj.update_skin(animation)
    showm.render()


##############################################################################
# Optional: `timeline.play()` auto plays the animations.


showm.add_timer_callback(True, 20, timer_callback)
scene.reset_camera()

interactive = False

if interactive:
    showm.start()

window.record(scene, out_path='viz_skinning.png', size=(900, 768))
