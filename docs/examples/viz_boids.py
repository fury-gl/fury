"""
==============================
Flocking Simulation with Boids
==============================

This example simulates a flock of "boids" (bird-oid objects) inside a box,
following Craig Reynolds' three local rules:

* **separation**: steer away from boids that are too close,
* **alignment**: steer toward the average heading of the neighbors,
* **cohesion**: steer toward the average position of the neighbors.

Each boid only sees the boids around it, yet groups form and travel together.
The gray spheres are obstacles that the boids avoid, and the large yellow cones
are attractors that pull nearby boids in. Boids slowly adopt the color of their
neighbors. Click on a boid to enlarge it and display its state.
"""

###############################################################################
# First, the imports.

import numpy as np

from fury import actor, ui, window
from fury.swarm import Swarm, box_edges, vec2vec_rotmat

###############################################################################
# The simulation state lives in a :class:`fury.swarm.Swarm`. Boids, obstacles
# and attractors start at random positions in the central part of the box,
# with random velocities.
#
# The boids move ``speed = 1`` unit per unit of simulation time. We render 60
# frames per second and advance the simulation by ``time_scale`` units of time
# per second, so each frame advances it by ``time_scale / fps``. Lower
# ``time_scale`` to slow the boids down without making the animation choppy.

rng = np.random.default_rng(42)
num_boids = 150
num_obstacles = 3
num_attractors = 2
box_size = np.array([100.0, 100.0, 100.0])
fps = 60
time_scale = 10.0


def random_positions(n):
    """Return ``n`` random positions in the central part of the box."""
    return box_size * (rng.random((n, 3)) - 0.5) * 0.6


swarm = Swarm(
    random_positions(num_boids),
    rng.random((num_boids, 3)) - 0.5,
    box_size=box_size,
    obstacle_positions=random_positions(num_obstacles),
    obstacle_velocities=(rng.random((num_obstacles, 3)) - 0.5) * 0.4,
    obstacle_radius=5.0,
    attractor_positions=random_positions(num_attractors),
    attractor_velocities=(rng.random((num_attractors, 3)) - 0.5) * 0.8,
    attractor_radius=3.0,
)

###############################################################################
# Pointing the cones along the velocities
# =======================================
#
# ``actor.cone`` stores all the cones in a single mesh, the vertices of cone
# ``i`` following those of cone ``i - 1``. The cone primitive points along the
# x axis. Each frame, we rotate this template cone toward every boid's
# velocity with ``vec2vec_rotmat`` and move it to the boid's position. NumPy
# does it for all the boids at once, without any Python loop.


def glyph_vertices(template, centers, directions, scales=1.0):
    """Return the vertices of copies of ``template`` placed at ``centers``."""
    rotations = vec2vec_rotmat((1, 0, 0), directions)
    vertices = np.einsum("nij,vj->nvi", rotations, template)
    vertices *= np.reshape(scales, (-1, 1, 1))
    return (vertices + centers[:, None, :]).reshape(-1, 3)


def make_glyphs(count, colors, height, radius):
    """Create ``count`` cones and return the actor and the template cone."""
    cones = actor.cone(
        np.zeros((count, 3)),
        colors=colors,
        height=height,
        radii=radius,
        directions=(1, 0, 0),
    )
    template = cones.geometry.positions.data[
        : len(cones.geometry.positions.data) // count
    ]
    return cones, template.copy()


boid_colors = rng.random((num_boids, 3))
boids, boid_template = make_glyphs(num_boids, boid_colors, height=2.0, radius=0.6)
attractors, attractor_template = make_glyphs(
    num_attractors, (1.0, 0.85, 0.1), height=6.0, radius=2.0
)

obstacles = [
    actor.sphere(np.zeros((1, 3)), colors=(0.6, 0.6, 0.6), radii=swarm.obstacle_radius)
    for _ in range(num_obstacles)
]

box = actor.line(box_edges(box_size), colors=(1.0, 0.5, 0.0))
# ``actor.line`` moves the line to its first point, but the edge coordinates
# are already absolute, so we put the box back at the origin.
box.local.position = (0.0, 0.0, 0.0)

info = ui.TextBlock2D(text="", position=(20, 20), font_size=16, dynamic_bbox=True)

scene = window.Scene()
scene.add(boids, attractors, box, info, *obstacles)

###############################################################################
# Picking a boid
# ==============
#
# Clicking on the boids mesh gives the index of the triangle under the mouse.
# Every cone has the same number of triangles, so the boid index follows.
# Clicking on a boid again releases it.

faces_per_boid = len(boids.geometry.indices.data) // num_boids
verts_per_boid = len(boid_template)
selected = np.zeros(num_boids, dtype=bool)
state = {"step": 0, "picked": None}


def on_pick(event):
    """Toggle the selection of the clicked boid."""
    boid = event.pick_info["face_index"] // faces_per_boid
    selected[boid] = not selected[boid]
    state["picked"] = boid if selected[boid] else None


boids.add_event_handler(on_pick, "pointer_down")

###############################################################################
# Sharing colors
# ==============
#
# At each step, a few boids take the color of a random neighbor, so the
# colors spread through the groups. ``swarm.neighbors[i, j]`` tells whether
# boid ``j`` was a neighbor of boid ``i`` in the last step.


def share_colors():
    """Let a few boids adopt the color of one of their neighbors."""
    neighbors = swarm.neighbors
    # About 6% of the boids per unit of simulation time.
    adopt_chance = 0.06 * time_scale / fps
    adopters = neighbors.any(axis=1) & (rng.random(num_boids) < adopt_chance)
    chosen = (rng.random(neighbors.shape) * neighbors).argmax(axis=1)
    boid_colors[adopters] = boid_colors[chosen[adopters]]


###############################################################################
# Updating the scene
# ==================
#
# After each simulation step, we write the new vertices and colors into the
# GPU buffers of the actors and flag the buffers as modified.


def draw():
    """Copy the simulation state into the actors."""
    positions = boids.geometry.positions
    positions.data[:] = glyph_vertices(
        boid_template,
        swarm.positions,
        swarm.velocities,
        np.where(selected, 2.5, 1.0),
    )
    positions.update_full()

    colors = boids.geometry.colors
    colors.data[:, :3] = np.repeat(boid_colors, verts_per_boid, axis=0)
    colors.update_full()

    positions = attractors.geometry.positions
    positions.data[:] = glyph_vertices(
        attractor_template, swarm.attractor_positions, swarm.attractor_velocities
    )
    positions.update_full()

    for sphere, center in zip(obstacles, swarm.obstacle_positions, strict=True):
        sphere.local.position = center

    text = f"Step {state['step']}\nClick on a boid to select it"
    boid = state["picked"]
    if boid is not None:
        text += (
            f"\n\nBoid {boid}"
            f"\nposition {np.round(swarm.positions[boid], 1)}"
            f"\nheading {np.round(swarm.velocities[boid], 2)}"
            f"\nneighbors {swarm.neighbors[boid].sum()}"
        )
    info.message = text


def update():
    """Advance the simulation by one step and redraw it."""
    swarm.step(dt=time_scale / fps)
    share_colors()
    state["step"] += 1
    draw()


###############################################################################
# Finally, we place the camera, draw the initial state and run the simulation
# ``fps`` times per second.

show_m = window.ShowManager(scene=scene, size=(1000, 800), title="FURY Boids")

camera = show_m.screens[0].camera
camera.local.position = (140.0, 100.0, 190.0)
camera.look_at((0.0, 0.0, 0.0))

draw()
show_m.register_callback(update, 1 / fps, True, "BoidsSimulation")
show_m.start()
