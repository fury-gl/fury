"""
==========================================
Electromagnetic Wave Propagation Animation
==========================================
A linearly polarized sinusoidal electromagnetic wave, propagating in the
direction -x (towards the origin) through a homogeneous, isotropic,
dissipationless medium.

The electric field (blue line) oscillates in the ±z-direction, and the
orthogonal magnetic field (red line) oscillates in phase in the ±y-direction.
"""

###############################################################################
# Importing necessary modules
import numpy as np

from fury import actor, ui, window

###############################################################################
# Let's define some variable and their description:
#
# * **npoints**: int
#       Number of spatial data sample points along the propagation axis.
#       Higher values yield smoother curves. (default = 800)
# * **wavelength**: float
#       The physical distance between successive wave crests. (default = 2.0)
# * **wavenumber**: float
#       Spatial frequency of the wave, defined as 2 * pi / wavelength.
# * **angular_frq**: float
#       Temporal frequency dictating rotation speed over time. (default = 1.0)
# * **phase_angle**: float
#       Initial angular displacement shift parameter at t=0. (default = 0.002)

npoints = 800
wavelength = 2.0
wavenumber = 2.0 * np.pi / wavelength
angular_frq = 1.0
phase_angle = 0.002

###############################################################################
# Generate the spatial coordinate arrays and initialize the animation state.

state = {"time": 0.0, "dt": 0.05}
x_vals = np.linspace(-3.0, 3.0, npoints)


###############################################################################
# Define helper functions to calculate magnetic and electric field coordinates.
def update_coordinates_m_field(x, t):
    """Calculate coordinates for the Magnetic Field oscillating in Y."""
    y = np.sin(wavenumber * x - angular_frq * t + phase_angle)
    return np.vstack([x, y, np.zeros_like(x)]).T


def update_coordinates_e_field(x, t):
    """Calculate coordinates for the Electric Field oscillating in Z."""
    z = np.sin(wavenumber * x - angular_frq * t + phase_angle)
    return np.vstack([x, np.zeros_like(x), z]).T


###############################################################################
# Creating a scene object and configuring the viewport background canvas.

scene = window.Scene()
scene.background = (0.1, 0.1, 0.15)


###############################################################################
# Create an arrow actor representing the direction of propagation.

centers = np.array([[3.0, 0.0, 0.0]])
directions = np.array([[-1.0, 0.0, 0.0]])
heights = np.array([6.4])

arrow_actor = actor.arrow(
    centers=centers,
    directions=directions,
    colors=(1.0, 1.0, 0.0),
    height=heights,
    resolution=20,
    tip_length=0.06,
    tip_radius=0.012,
    shaft_radius=0.005,
)
scene.add(arrow_actor)


###############################################################################
# Generate and add the initial magnetic field line actor to the scene.

pts_m = update_coordinates_m_field(x_vals, 0.0)
wave_actor1 = actor.line([pts_m], colors=(1.0, 0.0, 0.0), material="basic")
wave_actor1.local.position = (
    0.0,
    0.0,
    0.0,
)
scene.add(wave_actor1)


###############################################################################
# Generate and add the initial electric field line actor to the scene.

pts_e = update_coordinates_e_field(x_vals, 0.0)
wave_actor2 = actor.line([pts_e], colors=(0.0, 0.0, 1.0), material="basic")
wave_actor2.local.position = (
    0.0,
    0.0,
    0.0,
)
scene.add(wave_actor2)


###############################################################################
# Initializing 2D Text Block overlays to display the fields explanation legend.

hud_legend = ui.TextBlock2D(
    text="Electromagnetic Wave Propagation\n"
    "Red Line: Magnetic Field (Y-axis) | Blue Line: Electric Field (Z-axis)",
    position=(30, 30),
    font_size=16,
    color=(0.9, 0.9, 0.95),
    bold=True,
    dynamic_bbox=True,
)
scene.add(hud_legend)


###############################################################################
# Initializing show_m and camera parameters
#
# Low-level vertex changes are executed dynamically on each tick inside the
# callback loop by mutating the existing point data buffer memory arrays using
# .data[:, :] and calling .update_full() to trigger clean GPU updates.

show_m = window.ShowManager(
    scene=scene, size=(1024, 768), title="Fury Electromagnetic Wave"
)

camera = show_m.screens[0].camera
camera.local.position = (-6.0, 5.0, -10.0)
camera.look_at((0.0, 0.0, 0.0))


def update_playback_frame():
    state["time"] += state["dt"]
    t = state["time"]

    # # Zero-allocation buffer array mutation for the magnetic field component
    m_pts = update_coordinates_m_field(x_vals, t)
    wave_actor1.geometry.positions.data[:, :] = m_pts.astype(np.float32)
    wave_actor1.geometry.positions.update_full()

    # # Zero-allocation buffer array mutation for the electric field component
    e_pts = update_coordinates_e_field(x_vals, t)
    wave_actor2.geometry.positions.data[:, :] = e_pts.astype(np.float32)
    wave_actor2.geometry.positions.update_full()


###############################################################################
# Run every 30 milliseconds

show_m.register_callback(update_playback_frame, 0.03, True, "WavePropagationLoop")
show_m.start()
