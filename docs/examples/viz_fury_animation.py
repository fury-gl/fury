"""
======================================
Keyframe animation: Camera and opacity
======================================

Camera and opacity keyframe animation explained in this tutorial.
"""

import numpy as np

import fury
from fury.motion import Animation, CameraAnimation, Timeline, cubic_spline_interpolator
from fury.data import fetch_viz_cubemaps, read_viz_cubemap
from fury.io import load_cube_map_texture

###############################################################################
# The Plan
# ========
#
# The plan here is to animate, scale, and translate many spheres (count set by
# ``n_spheres``) randomly using a single sphere actor animated entirely on the
# GPU, then show ``FURY`` text at the end.

###############################################################################
# Number of spheres to animate. Change this value to animate more or fewer
# spheres; everything below adapts to it.

n_spheres = 100000

###############################################################################
# Let's fetch and load a skybox texture.

fetch_viz_cubemaps()
texture_files = read_viz_cubemap("skybox")
cube_map = load_cube_map_texture(texture_files)

scene = fury.window.Scene(skybox=cube_map)

showm = fury.window.ShowManager(
    scene=scene, size=(900, 768), pixel_ratio=2, show_fps=True
)

###############################################################################
# Creating the main ``Timeline``
# ==============================
#
# Here we create a ``Timeline`` so that it can control all animations from one
# place. A playback panel is attached to provide interactive playback controls.

timeline = Timeline(playback_panel=True, loop=True)

###############################################################################
# Creating an actor for visualization.

plane = fury.actor.box(
    np.array([[0, 0, 0]]),
    colors=np.array([[1, 1, 1]]),
    scales=np.array([[20, 0.2, 20]]),
)

###############################################################################
# Creating ``FURY`` text
# ======================

fury_text = fury.actor.text(
    text="FURY", position=(0, 15, 0), font_size=5, anchor="middle-center"
)

###############################################################################
# Creating an ``Animation`` to animate the opacity of ``fury_text``.

text_anim = Animation(actors=fury_text, loop=False)

###############################################################################
# Opacity is set to 0 at time 29 and to 1 at time 35. Linear interpolation is
# used by default.

text_anim.set_opacity(29, 0.0)
text_anim.set_opacity(35, 1.0)
timeline.add_animation(text_anim)

###############################################################################
# Creating and animating spheres on the GPU
# =========================================
#
# One impostor sphere actor holds all ``n_spheres`` spheres. Keyframes for
# every sphere are uploaded to the GPU once; each frame only the current time
# is sent and a compute shader writes the sphere centers and radii into the
# actor's vertex buffers.

base_radii = np.random.random(n_spheres) * 0.5 + 0.1

sphere_actor = fury.actor.sphere(
    np.zeros((n_spheres, 3)),
    colors=np.random.random((n_spheres, 3)),
    radii=base_radii,
)

###############################################################################
# Each keyframe value holds the state of ALL spheres at that time:
# ``(n_spheres, 3)`` centers and ``(n_spheres,)`` radii.

keyframe_times = range(0, 50, 2)
center_keyframes = {
    t: np.random.random((n_spheres, 3)) * 30 - np.array([15, 0, 15])
    for t in keyframe_times
}
radius_keyframes = {t: base_radii * np.random.random(n_spheres) for t in keyframe_times}

animation = Animation(actors=sphere_actor)
animation.set_gpu_keyframes("centers", center_keyframes, interpolation="cubic")
animation.set_gpu_keyframes("radii", radius_keyframes)
timeline.add_animation(animation)

###############################################################################
# Animating the camera
# ====================
#
# Camera animations are represented with ``CameraAnimation``. Three properties
# can control the camera: position, focal position, and view-up direction.

camera_anim = CameraAnimation(loop=False)
timeline.add_animation(camera_anim)

camera_positions = {
    0: np.array([3, 3, 3]),
    4: np.array([50, 25, -40]),
    7: np.array([-50, 50, -40]),
    10: np.array([-25, 25, 20]),
    14: np.array([0, 16, 25]),
    20: np.array([0, 14.5, 20]),
}

camera_focal_positions = {
    15: np.array([0, 0, 0]),
    20: np.array([3, 9, 5]),
    23: np.array([7, 5, 3]),
    25: np.array([-2, 9, -6]),
    27: np.array([0, 16, 0]),
    31: np.array([0, 14.5, 0]),
}

camera_anim.set_position_keyframes(camera_positions)
camera_anim.set_focal_keyframes(camera_focal_positions)

###############################################################################
# Change camera position and focal interpolators.

camera_anim.set_position_interpolator(cubic_spline_interpolator)
camera_anim.set_focal_interpolator(cubic_spline_interpolator)

###############################################################################
# Add non-animated actors to the scene.

scene.add(plane)

###############################################################################
# Add the timeline to the ``ShowManager`` for interactive playback.

showm.add_animation(timeline)

###############################################################################
# Recording the animation
# =======================
#

record_animation = False

if record_animation:
    timeline.record("viz_fury_animation.mp4", fps=30)
else:
    ###########################################################################
    # The ShowManager must go on!

    showm.start()
