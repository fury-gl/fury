"""
===============================================
Motion of a Charged Particle in Combined Fields
===============================================
A charged particle follows a curved path in a magnetic field.
In an electric field, the particle tends to accelerate in a direction
parallel/antiparallel to the electric field depending on the nature of
charge on the particle. In a combined electric and magnetic field,
the particle moves along a helical path.
"""

###############################################################################
# Import the required libraries.
import numpy as np

from fury import actor, ui, window

###############################################################################
# Let's define some variable and their description:
#
# * **radius_particle**: float
#       Physical radius size of the rendered particle sphere object. (default = 0.08)
# * **initial_velocity**: float
#       The starting linear forward speed along the field alignment axis.
# * **acc**: float
#       Constant acceleration rate driven by the parallel electric field force.
# * **incre_time**: float
#       Differential step variable by which execution time clock parameters tick up.
# * **angular_frq**: float
#       Cyclotron angular velocity frequency driving the orbital rotation rate.

radius_particle = 0.08
initial_velocity = 0.09
acc = 0.004
incre_time = 0.09
angular_frq = 0.1
phase_angle = 0.002

state = {"time": 0.0, "step_count": 0, "path_history": [], "line_actor": None}

###############################################################################
# Creating a scene object and configuring the viewport background canvas.

scene = window.Scene()
scene.background = (0.08, 0.08, 0.12)

###############################################################################
# Create an arrow actor representing the direction of the combined fields.

centers = np.array([[0, 0.0, 0.0]], dtype=np.float32)
directions = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
arrow_actor = actor.arrow(
    centers=centers,
    directions=directions,
    colors=(0.2, 0.5, 1.0),
    height=4.5,
    resolution=20,
    tip_length=0.15,
    tip_radius=0.12,
    shaft_radius=0.05,
)
scene.add(arrow_actor)

###############################################################################
# Calculate the initial position of the particle and spawn the charge sphere actor.

x_init = initial_velocity * 0.0 + 0.5 * acc * (0.0**2)
y_init = np.sin(angular_frq * 0.0 + phase_angle)
z_init = np.cos(angular_frq * 0.0 + phase_angle)

charge_actor = actor.sphere(
    centers=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
    radii=np.array([radius_particle], dtype=np.float32),
    colors=np.array([[1.0, 0.2, 0.2, 1.0]], dtype=np.float32),
)

# Relocate the sphere inside world coords using localized position vector transforms.
charge_actor.local.position = (x_init, y_init, z_init)
scene.add(charge_actor)

state["path_history"].append([x_init, y_init, z_init])

###############################################################################
# Add a 2D text overlay to display the legend.
hud_legend = ui.TextBlock2D(
    text="Charged Particle Lorentz Motion\n"
    "Red Sphere: Positively Charged Particle | Cyan Trail: Helical Trajectory\n"
    "Blue Arrow: Electric & Magnetic Fields (Centered along X-Axis)",
    position=(30, 30),
    font_size=15,
    color=(0.95, 0.95, 1.0),
    bold=True,
    dynamic_bbox=True,
)
scene.add(hud_legend)

###############################################################################
# Initialize the ShowManager, position the virtual camera, and register the update loop.
show_m = window.ShowManager(
    scene=scene, size=(1024, 768), title="Fury Lorentz Field Helical Motion"
)

camera = show_m.screens[0].camera
camera.local.position = (10.0, 12.5, 19.0)
camera.look_at((3.0, 0.0, 0.0))


def update_helical_motion():
    """Calculate helical coordinates and update the line and sphere positions."""
    if state["step_count"] >= 350:
        show_m.cancel_callback("HelicalMotionLoop")
        return

    state["time"] += incre_time
    t = state["time"]

    x = initial_velocity * t + 0.5 * acc * (t**2)
    y = np.sin(10.0 * angular_frq * t + phase_angle)
    z = np.cos(10.0 * angular_frq * t + phase_angle)

    charge_actor.local.position = (x, y, z)

    state["path_history"].append([x, y, z])

    if state["line_actor"] is not None:
        scene.remove(state["line_actor"])

    pts_array = np.array(state["path_history"], dtype=np.float32)
    state["line_actor"] = actor.line(
        [pts_array], colors=(0.0, 1.0, 1.0), material="basic"
    )

    state["line_actor"].local.position = (0.0, 0.0, 0.0)

    scene.add(state["line_actor"])

    state["step_count"] += 1


###############################################################################
# Run every 15 milliseconds

show_m.register_callback(update_helical_motion, 0.015, True, "HelicalMotionLoop")
show_m.start()
