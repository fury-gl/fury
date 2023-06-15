"""
========
Fractals
========

Fractals are geometric structures that are self-similar at any scale. These
structures are easy to generate using recursion. In this demo, we'll be
implementing the following fractals:

- Sierpinski Tetrahedron or Tetrix
- Menger Sponge
- Moseley Snowflake

Let's begin by importing some necessary modules. We need ``fury.primitive`` to
avoid having to hardcode the geometry of a tetrahedron and a cube.
``fury.utils`` also contains a ``repeat_primitive`` function which we will use
for this demo.
"""

import math
import numpy as np
from fury import window, primitive, utils, ui

###############################################################################
# Before we create our first fractal, let's set some ground rules for us to
# work with.
#
# 1. Instead of creating a new actor to represent each primitive of the
# fractal, we will compute the centers of each primitive and draw them at once
# using ``repeat_primitive()``.
#
# 2. How many primitives do we need? For each fractal, we define a depth which
# will prevent infinite recursion. Assuming we have a depth of :math:`N`, and
# at each level the shape is divided into :math:`k` smaller parts, we will need
# :math:`k^{N}` primitives to represent the fractal.
#
# 3. Ideally, we want to allocate the array of centers upfront. To achieve
# this, we can use the method of representing a binary tree in an array, and
# extend it to work with k-ary trees (formulas for the same can be found
# `here`_). In this scheme of representation, we represent every primitive as a
# node, and each sub-primitive as a child node. We can also skip storing the
# first :math:`\frac{k^{N} - 1}{k - 1} + 1` entries as we only need to render
# the leaf nodes. This allows us to create an array of exactly the required
# size at the start, without any additional overhead.
#
# .. _here: https://book.huihoo.com/data-structures-and-algorithms-with-object-oriented-design-patterns-in-c++/html/page356.html # noqa
#
# -----------------------------------------------------------------------------

###############################################################################
# The tetrix is a classic 3d fractal, a natural three-dimensional extension of
# the Sierpinski Triangle. At each level, we need to calculate the new centers
# for the next level. We can use the vertices of a tetrahedron as the offsets
# for the new centers, provided that the tetrahedron is centered at the origin
# (which is the case here).


def tetrix(N):
    centers = np.zeros((4 ** N, 3))

    # skipping non-leaf nodes (see above)
    offset = (4 ** N - 1) // 3 + 1

    # just need the vertices
    U, _ = primitive.prim_tetrahedron()

    def gen_centers(depth, pos, center, dist):
        if depth == N:
            centers[pos - offset] = center
        else:
            idx = 4 * (pos - 1) + 2
            for i in range(4):
                # distance gets halved at each level
                gen_centers(depth + 1, idx + i, center + dist * U[i], dist / 2)

    # the division by sqrt(6) is to ensure correct scale
    gen_centers(0, 1, np.zeros(3), 2 / (6 ** 0.5))

    vertices, faces = primitive.prim_tetrahedron()

    # primitive is scaled down depending on level
    vertices /= 2 ** (N - 1)

    # compute some pretty colors
    bounds_min, bounds_max = np.min(centers, axis=0), np.max(centers, axis=0)
    colors = (centers - bounds_min) / (bounds_max - bounds_min)

    vertices, triangles, colors, _ = primitive.repeat_primitive(
        centers=centers, colors=colors, vertices=vertices, faces=faces
    )
    return utils.get_actor_from_primitive(vertices, triangles, colors)

###############################################################################
# For a Menger Sponge, each cube is divided into 27 smaller cubes, and we skip
# some of them (face centers, and the center of the cube). This means that on
# every level we get 20 new cubes.
#
# Here, to compute the points of each new center, we start at a corner cube's
# center and add the offsets to each smaller cube, scaled according to the
# level.


def sponge(N):
    centers = np.zeros((20 ** N, 3))
    offset = (20 ** N - 1) // 19 + 1

    # these are the offsets of the new centers at the next level of recursion
    # each cube is divided into 20 smaller cubes for a snowflake
    V = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 2],
                  [0, 2, 0], [0, 2, 1], [0, 2, 2], [1, 0, 0], [1, 0, 2],
                  [1, 2, 0], [1, 2, 2], [2, 0, 0], [2, 0, 1], [2, 0, 2],
                  [2, 1, 0], [2, 1, 2], [2, 2, 0], [2, 2, 1], [2, 2, 2]])

    def gen_centers(depth, pos, center, dist):
        if depth == N:
            centers[pos - offset] = center
        else:
            # we consider a corner cube as our starting point
            start = center - np.array([1, 1, 1]) * dist ** 0.5
            idx = 20 * (pos - 1) + 2

            # this moves from the corner cube to each new cube's center
            for i in range(20):
                # each cube is divided into 27 cubes so side gets divided by 3
                gen_centers(depth + 1, idx + i, start + V[i] * dist, dist / 3)

    gen_centers(0, 1, np.zeros(3), 1 / 3)

    vertices, faces = primitive.prim_box()
    vertices /= 3 ** N

    bounds_min, bounds_max = np.min(centers, axis=0), np.max(centers, axis=0)
    colors = (centers - bounds_min) / (bounds_max - bounds_min)

    vertices, triangles, colors, _ = primitive.repeat_primitive(
        centers=centers, colors=colors, vertices=vertices, faces=faces
    )
    return utils.get_actor_from_primitive(vertices, triangles, colors)


###############################################################################
# A snowflake is exactly the same as above, but we skip different cubes
# (corners and center). I think this looks quite interesting, and it is
# possible to see the Koch snowflake if you position the camera just right.

def snowflake(N):
    centers = np.zeros((18 ** N, 3))
    offset = (18 ** N - 1) // 17 + 1
    V = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 1],
                  [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 2],
                  [1, 2, 0], [1, 2, 1], [1, 2, 2], [2, 0, 1], [2, 1, 0],
                  [2, 1, 1], [2, 1, 2], [2, 2, 1]])

    def gen_centers(depth, pos, center, side):
        if depth == N:
            centers[pos - offset] = center
        else:
            start = center - np.array([1, 1, 1]) * side ** 0.5
            idx = 18 * (pos - 1) + 2
            for i in range(18):
                gen_centers(depth + 1, idx + i, start + V[i] * side, side / 3)

    gen_centers(0, 1, np.zeros(3), 1 / 3)

    vertices, faces = primitive.prim_box()
    vertices /= 3 ** N

    bounds_min, bounds_max = np.min(centers, axis=0), np.max(centers, axis=0)
    colors = (centers - bounds_min) / (bounds_max - bounds_min)

    vertices, triangles, colors, _ = primitive.repeat_primitive(
        centers=centers, colors=colors, vertices=vertices, faces=faces
    )
    return utils.get_actor_from_primitive(vertices, triangles, colors)


###############################################################################
# Now that we have the functions to generate fractals, we can start setting up
# the Scene and ShowManager.

scene = window.Scene()
showmgr = window.ShowManager(scene, "Fractals", (800, 800), reset_camera=True)

###############################################################################
# These values are what work nicely on my machine without lagging. If you have
# a powerful machine, you could bump these up by around 2-3.

fractals = [tetrix(6), sponge(3), snowflake(3)]

###############################################################################
# We want to be able to switch between the three fractals. To achieve this
# we'll create a RadioButton and register a callback which will remove existing
# fractals and add the selected one. This also resets the camera.

options = {
    "Tetrix": 0,
    "Sponge": 1,
    "Snowflake": 2,
}

shape_chooser = ui.RadioButton(options.keys(), padding=10, font_size=16,
                               checked_labels=["Tetrix"], position=(10, 10))


def choose_shape(radio):
    showmgr.scene.rm(*fractals)
    showmgr.scene.add(fractals[options[radio.checked_labels[0]]])
    showmgr.scene.reset_camera()


shape_chooser.on_change = choose_shape

# selected at start
showmgr.scene.add(fractals[0])
showmgr.scene.add(shape_chooser)

###############################################################################
# Let's add some basic camera movement to make it look a little interesting.
# We can use a callback here to update a counter and calculate the camera
# positions using the counter. ``sin`` and ``cos`` are used here to make smooth
# looping movements.

counter = 0


def timer_callback(_obj, _event):
    global counter
    counter += 1
    showmgr.scene.azimuth(math.sin(counter * 0.01))
    showmgr.scene.elevation(math.cos(counter * 0.01) / 4)
    showmgr.render()


showmgr.add_timer_callback(True, 20, timer_callback)

###############################################################################
# Finally, show the window if running in interactive mode or render to an image
# otherwise. This is needed for generating the documentation that you are
# reading.

interactive = False
if interactive:
    showmgr.start()
else:
    window.record(showmgr.scene, out_path="fractals.png", size=(800, 800))
