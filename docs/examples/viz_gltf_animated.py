"""
================================
Visualizing a animated glTF file
================================

In this tutorial, we will show how to display an animated glTF file in a
scene.

A glTF file can carry animation clips: keyframe tracks that drive node
transforms, the joint matrices of a skeleton, or morph target weights. FURY
exposes each clip as a ``GLTFAnimation``, which a ``Timeline`` plays alongside
any other animation in the scene.
"""

#########################################################################################
# Import the required libraries.
import fury
from fury.gltf import glTF

#########################################################################################
# Retrieve the glTF model. ``Fox`` is rigged and carries three named clips.
fury.data.fetch_gltf(name="Fox", mode="glTF")
filename = fury.data.read_viz_gltf("Fox")

#########################################################################################
# Initialize the glTF object and inspect the clips it carries.
#
# ``animations`` is keyed by clip name. Clips the file leaves unnamed are keyed
# ``anim_0``, ``anim_1`` and so on, in the order they are declared.
gltf_obj = glTF(filename)

print(f"{filename} carries the clips {list(gltf_obj.animations)}")

#########################################################################################
# Pick one clip and hand it to a ``Timeline``.
#
# Fox's three clips drive the same skeleton, so they are alternatives rather
# than layers: play one at a time. For a file whose clips are meant to run
# together, ``gltf_obj.main_animation()`` returns a ``Timeline`` holding all of
# them at once.
#
# The playback panel gives interactive play, pause, loop and speed controls.
animation = gltf_obj.animations["Walk"]
timeline = fury.motion.Timeline(animations=animation, playback_panel=True)

#########################################################################################
# Set up the scene. The actors do not need to be added separately: registering
# the Timeline below adds the imported scene along with it.
scene = fury.window.Scene(background=(0.05, 0.05, 0.08))
show_m = fury.window.ShowManager(
    scene=scene, size=(1280, 720), title="FURY glTF Animation"
)

#########################################################################################
# Register the timeline with the ShowManager, which updates it every frame. The
# camera frames the scene and carries its own light, and dragging orbits it.
show_m.add_animation(timeline)

#########################################################################################
# Start the rendering loop.
show_m.start()
