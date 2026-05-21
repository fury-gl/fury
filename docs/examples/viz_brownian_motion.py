"""
=============================
Brownian Motion 3D Simulation
=============================

Brownian motion is the random motion of particles suspended in a medium.
This animation plots the 3D trajectories of 20 random paths inside a
containment box.
"""

###############################################################################
# Import the required libraries and define the particle and simulation parameters.
import numpy as np
from scipy.stats import norm
from fury import actor, ui, window

NUM_PARTICLES = 20
NUM_TOTAL_STEPS = 300
DELTA_SPEED = 1.8
TIME_STEP = 5.0 / NUM_TOTAL_STEPS
BOX_SCALE = 20.0

###############################################################################
# Generate the random 3D trajectory data and initialize the animation state.
state = {
    "counter_step": 1,
    "trajectories": [
        np.zeros((NUM_TOTAL_STEPS, 3), dtype=np.float32) for _ in range(NUM_PARTICLES)
    ],
    "colors": [np.random.rand(3).astype(np.float32) for _ in range(NUM_PARTICLES)],
    "line_actor": None,
}

scale_factor = DELTA_SPEED**2 * TIME_STEP
for p_idx in range(NUM_PARTICLES):
    current_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    state["trajectories"][p_idx][0] = current_pos

    for step in range(1, NUM_TOTAL_STEPS):
        displacement = norm.rvs(scale=scale_factor, size=3).astype(np.float32)
        current_pos += displacement
        state["trajectories"][p_idx][step] = current_pos

###############################################################################
# Initialize the 3D scene viewport and add a semi-transparent containment box.
scene = window.Scene()
scene.background = (0.95, 0.95, 0.98)

container_actor = actor.box(
    centers=np.array([[0.0, 0.0, 0.0]]),
    colors=np.array([[0.4, 0.6, 0.5, 0.2]], dtype=np.float32),
    scales=np.array([[BOX_SCALE, BOX_SCALE, BOX_SCALE]]),
)
scene.add(container_actor)

###############################################################################
# Add a 2D text overlay to display the legend.
tb = ui.TextBlock2D(
    text="Brownian Motion 3D",
    position=(30, 30),
    font_size=20,
    color=(0.1, 0.1, 0.1),
    bold=True,
    dynamic_bbox=True,
)
scene.add(tb)


###############################################################################
# Define the animation callback to update the active 3D trace lines dynamically.
def update_brownian_motion(show_m):
    """Calculate active coordinate traces sequentially across execution frames."""
    step = state["counter_step"]
    if step >= NUM_TOTAL_STEPS:
        show_m.cancel_callback("BrownianSimulationLoop")
        return

    if state["line_actor"] is not None:
        scene.remove(state["line_actor"])

    active_paths = [state["trajectories"][i][:step] for i in range(NUM_PARTICLES)]

    state["line_actor"] = actor.line(
        active_paths, colors=state["colors"], material="basic"
    )
    scene.add(state["line_actor"])

    state["counter_step"] += 1


###############################################################################
# Initialize the ShowManager, position the virtual camera, and register the update loop.
if __name__ == "__main__":
    show_m = window.ShowManager(
        scene=scene, size=(800, 800), title="Fury Brownian Motion"
    )

    camera = show_m.screens[0].camera
    camera.local.position = (15.0, 20.0, 35.0)
    camera.look_at((0.0, 0.0, 0.0))

    show_m.register_callback(
        lambda: update_brownian_motion(show_m), 0.03, True, "BrownianSimulationLoop"
    )

    show_m.start()
