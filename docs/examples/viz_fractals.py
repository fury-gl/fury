"""
===================
3D Fractal Explorer
===================
Fractals are geometric structures that are self-similar at any scale.
This interactive demonstration implements a real-time explorer using
recursion to generate three-dimensional mathematical fractals.
"""

###############################################################################
# Importing necessary modules
import numpy as np

from fury import actor, primitive, ui, window
from fury.lib import EventType

###############################################################################
# Let's define some variable and their description:
#
# * **state**: dict
#       A global container tracking internal application statuses:
#       - `counter`: step index used for computing smooth orbital camera sweeps.
#       - `active_index`: pointer tracking which fractal asset is visible.
# * **fractals**: list
#       Collection containing the compiled high-performance surface actors.

state = {
    "counter": 0,
    "active_index": 0,
}


###############################################################################
# Define the generation function for the Sierpinski Tetrahedron.


def tetrix(N):
    """Generate a Sierpinski Tetrahedron/Tetrix with recursive geometry calculation."""
    centers = np.zeros((4**N, 3))
    offset = (4**N - 1) // 3 + 1
    U, _ = primitive.prim_tetrahedron()

    def gen_centers(depth, pos, center, dist):
        if depth == N:
            centers[pos - offset] = center
        else:
            idx = 4 * (pos - 1) + 2
            for i in range(4):
                gen_centers(depth + 1, idx + i, center + dist * U[i], dist / 2.0)

    gen_centers(0, 1, np.zeros(3), 2.0 / (6.0**0.5))

    vertices, faces = primitive.prim_tetrahedron()
    vertices /= 2.0 ** (N - 1)

    bounds_min, bounds_max = np.min(centers, axis=0), np.max(centers, axis=0)

    denom = bounds_max - bounds_min
    denom[denom == 0.0] = 1.0
    colors = (centers - bounds_min) / denom

    vertices, triangles, colors, _ = primitive.repeat_primitive(
        centers=centers, colors=colors, vertices=vertices, faces=faces
    )

    return actor.surface(
        vertices.astype(np.float32),
        faces=triangles.astype(np.int32),
        colors=colors.astype(np.float32),
    )


###############################################################################
# Define the generation function for the Menger Sponge.


def sponge(N):
    """Generate a Menger Sponge fractal using recursive subdivision."""
    centers = np.zeros((20**N, 3))
    offset = (20**N - 1) // 19 + 1

    V = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 1, 0],
            [0, 1, 2],
            [0, 2, 0],
            [0, 2, 1],
            [0, 2, 2],
            [1, 0, 0],
            [1, 0, 2],
            [1, 2, 0],
            [1, 2, 2],
            [2, 0, 0],
            [2, 0, 1],
            [2, 0, 2],
            [2, 1, 0],
            [2, 1, 2],
            [2, 2, 0],
            [2, 2, 1],
            [2, 2, 2],
        ]
    )

    def gen_centers(depth, pos, center, dist):
        if depth == N:
            centers[pos - offset] = center
        else:
            start = center - np.array([1.0, 1.0, 1.0]) * dist**0.5
            idx = 20 * (pos - 1) + 2
            for i in range(20):
                gen_centers(depth + 1, idx + i, start + V[i] * dist, dist / 3.0)

    gen_centers(0, 1, np.zeros(3), 1.0 / 3.0)

    vertices, faces = primitive.prim_box()
    vertices /= 3.0**N

    bounds_min, bounds_max = np.min(centers, axis=0), np.max(centers, axis=0)
    denom = bounds_max - bounds_min
    denom[denom == 0.0] = 1.0
    colors = (centers - bounds_min) / denom

    vertices, triangles, colors, _ = primitive.repeat_primitive(
        centers=centers, colors=colors, vertices=vertices, faces=faces
    )

    return actor.surface(
        vertices.astype(np.float32),
        faces=triangles.astype(np.int32),
        colors=colors.astype(np.float32),
    )


###############################################################################
# Define the generation function for the Moseley Snowflake.


def snowflake(N):
    """Generate a Moseley Snowflake fractal using iterative coordinate transforms."""
    centers = np.zeros((18**N, 3))
    offset = (18**N - 1) // 17 + 1

    V = np.array(
        [
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 0, 2],
            [1, 1, 0],
            [1, 1, 2],
            [1, 2, 0],
            [1, 2, 1],
            [1, 2, 2],
            [2, 0, 1],
            [2, 1, 0],
            [2, 1, 1],
            [2, 1, 2],
            [2, 2, 1],
        ]
    )

    def gen_centers(depth, pos, center, side):
        if depth == N:
            centers[pos - offset] = center
        else:
            start = center - np.array([1.0, 1.0, 1.0]) * side**0.5
            idx = 18 * (pos - 1) + 2
            for i in range(18):
                gen_centers(depth + 1, idx + i, start + V[i] * side, side / 3.0)

    gen_centers(0, 1, np.zeros(3), 1.0 / 3.0)

    vertices, faces = primitive.prim_box()
    vertices /= 3.0**N

    bounds_min, bounds_max = np.min(centers, axis=0), np.max(centers, axis=0)
    denom = bounds_max - bounds_min
    denom[denom == 0.0] = 1.0
    colors = (centers - bounds_min) / denom

    vertices, triangles, colors, _ = primitive.repeat_primitive(
        centers=centers, colors=colors, vertices=vertices, faces=faces
    )

    return actor.surface(
        vertices.astype(np.float32),
        faces=triangles.astype(np.int32),
        colors=colors.astype(np.float32),
    )


###############################################################################
# Pre-generate fractal models at start.

fractals = [tetrix(5), sponge(3), snowflake(3)]


###############################################################################
# Instantiate the 3D scene viewport and add the default fractal actor.
scene = window.Scene()
scene.background = (0.05, 0.05, 0.1)
scene.add(fractals[state["active_index"]])


###############################################################################
# Add a 2D text overlay to display the legend.
hud_legend = ui.TextBlock2D(
    text="",
    position=(30, 30),
    font_size=16,
    color=(0.95, 0.95, 1.0),
    bold=True,
    dynamic_bbox=True,
)
scene.add(hud_legend)


###############################################################################
# Define a utility function to update the HUD text contents.


def update_hud():
    """Modify the TextBlock2D values to match the currently selected fractal."""
    names = ["Sierpinski Tetrix", "Menger Sponge", "Moseley Snowflake"]
    text_content = (
        "Fury 3D Fractal Explorer\n"
        f"Active: {names[state['active_index']]}\n"
        "Press [1]: Tetrix | [2]: Sponge | [3]: Snowflake"
    )
    hud_legend.text = text_content
    hud_legend.message = text_content


# Run an initial update to display correct information.
update_hud()


###############################################################################
# Initialize the ShowManager, register keyboard events, and start the animation loop.
show_m = window.ShowManager(scene=scene, size=(1024, 768), title="Fury 3D Fractals")

# Position the camera to overlook the centered geometries.
camera = show_m.screens[0].camera
camera.local.position = (4.0, 3.0, 6.0)
camera.look_at((0.0, 0.0, 0.0))


def handle_key_event(event):
    """Map keyboard inputs 1, 2, and 3 to swap active fractal geometry actors."""
    key = event.key.lower()
    if key in ["1", "2", "3"]:
        new_index = int(key) - 1
        if new_index != state["active_index"]:
            scene.remove(fractals[state["active_index"]])
            state["active_index"] = new_index
            scene.add(fractals[new_index])

            camera.look_at((0.0, 0.0, 0.0))
            update_hud()
            show_m.render()


show_m.renderer.add_event_handler(handle_key_event, EventType.KEY_UP)


def update_playback_frame():
    """Rotate the camera orbital viewpoint around the origin every frame."""
    state["counter"] += 1
    angle = state["counter"] * 0.012
    r = 6.0

    x = r * np.sin(angle)
    z = r * np.cos(angle)
    y = 2.5 * np.sin(angle * 0.5)

    camera.local.position = (x, y, z)
    camera.look_at((0.0, 0.0, 0.0))


###############################################################################
# Run every 30 milliseconds

show_m.register_callback(update_playback_frame, 0.03, True, "FractalOrbitLoop")
show_m.start()
