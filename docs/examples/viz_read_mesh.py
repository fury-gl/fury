"""
==============
Reading a Mesh
==============

This example demonstrates how to fetch 3D mesh assets from the
``polyxios-data`` repository, read them into NumPy arrays with
:func:`fury.io.read_mesh`, and render them as colored surface actors. We load
two files of different formats side by side -- a ``.vtp`` surface and a ``.obj``
mesh -- to highlight that the same code handles both.

Polyxios is the lightweight, dependency-free mesh I/O backend used by FURY. It
reads and writes the common scientific surface formats (``.vtk``, ``.vtp``,
``.ply``, ``.obj`` and the VTK XML family) straight into NumPy arrays, and its
``fetch`` helper downloads and caches sample assets so tutorials and tests can
run without bundling large files.

``read_mesh`` always returns the same simple triple regardless of the source
format:

* ``vertices`` -- ``(N, 3)`` float32 point coordinates,
* ``faces`` -- ``(M, 3)`` int32 triangle indices (surfaces are triangulated
  for you), and
* ``colors`` -- ``(N, 3)`` float32 per-vertex RGB in ``[0, 1]`` (or ``None``).
"""

#########################################################################################
# Import the required libraries.
import numpy as np
import polyxios as px

from fury import actor, ui, window
from fury.io import read_mesh

#########################################################################################
# Define a small helper that reads a mesh and turns it into a FURY actor with
# :func:`fury.actor.surface`. It centers and normalizes the geometry and falls
# back to a height-based color gradient when the file has no colors.
#
# ``read_mesh`` returns the same ``(vertices, faces, colors)`` triple for every
# supported format, so the exact same code path loads ``.obj`` and ``.vtp``
# files alike.


def mesh_to_actor(file_path):
    """Read a mesh file and build a colored surface actor."""
    vertices, faces, colors = read_mesh(file_path)

    # Center on the origin and normalize the scale to roughly unit size.
    vertices = vertices - vertices.mean(axis=0)
    vertices = (vertices / np.max(np.abs(vertices))).astype(np.float32)

    # Use the file's colors when present, otherwise a height-based gradient.
    if colors is None:
        height = vertices[:, 1]
        t = ((height - height.min()) / (height.max() - height.min()))[:, None]
        low_color = np.array([0.10, 0.20, 0.70], dtype=np.float32)
        high_color = np.array([0.95, 0.75, 0.20], dtype=np.float32)
        colors = (low_color * (1.0 - t) + high_color * t).astype(np.float32)

    # ``actor.surface`` wraps the geometry, material and mesh creation for us,
    # turning the vertices, faces and per-vertex colors into a ready actor.
    surf = actor.surface(vertices, faces, colors=colors)
    return surf, len(vertices), len(faces)


#########################################################################################
# Fetch two assets from the ``polyxios-data`` release. ``fetch`` returns the
# absolute path to the locally cached file, downloading it on first use.
#
# * ``Human.vtp`` is a VTK XML PolyData surface that already carries per-vertex
#   colors.
# * ``stanford-bunny.obj`` is the classic Stanford bunny stored as a Wavefront
#   OBJ with plain geometry (no colors).
human_path = px.fetch("Human.vtp")
bunny_path = px.fetch("stanford-bunny.obj")

human_actor, human_nv, human_nf = mesh_to_actor(human_path)
bunny_actor, bunny_nv, bunny_nf = mesh_to_actor(bunny_path)

print(f"Human.vtp: {human_nv} vertices, {human_nf} faces")
print(f"stanford-bunny.obj: {bunny_nv} vertices, {bunny_nf} faces")

#########################################################################################
# ``Human.vtp`` is modeled lying along its Z axis, so by default it faces the
# camera end-on. Every FURY actor exposes transform helpers (``rotate``,
# ``translate``, ``scale``), so we stand it upright with a -90 degrees rotation
# about the X axis, mapping its head-to-toe axis to the vertical and apply 180 degrees
# so it faces the camera.
human_actor.rotate((-90, 180, 0))

#########################################################################################
# Place the two meshes side by side so both formats are visible at once.
human_actor.local.position = (-1.3, 0.0, 0.0)
bunny_actor.local.position = (1.3, 0.0, 0.0)

#########################################################################################
# Add a 3D text label beneath each mesh. ``actor.text`` accepts a list of
# strings with matching positions and returns a Group of 3D Text actors that
# live in the scene (unlike the 2D HUD overlay, these are part of the world and
# move with the camera).
labels_actor = actor.text(
    ["Human.vtp", "stanford-bunny.obj"],
    position=[(-1.3, -1.25, 0.0), (1.3, -1.25, 0.0)],
    colors=(0.9, 0.9, 0.95),
    font_size=0.18,
    anchor="top-center",
)

#########################################################################################
# Set up the 3D scene and add both mesh actors and their labels.
scene = window.Scene(background=(0.05, 0.05, 0.08))
scene.add(human_actor)
scene.add(bunny_actor)
scene.add(labels_actor)

#########################################################################################
# Add a 2D text overlay describing what is being shown. The same few lines of
# code load OBJ, PLY, VTK and VTP files because ``read_mesh`` normalizes every
# format to the same NumPy arrays.
info_text = (
    f"FURY x Polyxios mesh reader\n"
    f"Left:  Human.vtp  ({human_nv} verts, {human_nf} faces, file colors)\n"
    f"Right: stanford-bunny.obj  ({bunny_nv} verts, {bunny_nf} faces)\n"
    f"Both read via fury.io.read_mesh -> (vertices, faces, colors)\n"
    f"Supported: .vtk  .vtp  .ply  .obj  (auto-detected by extension)"
)

hud_label = ui.TextBlock2D(
    text=info_text,
    position=(20, 20),
    font_size=16,
    color=(0.9, 0.9, 0.95),
    bold=False,
    dynamic_bbox=True,
)
scene.add(hud_label)

#########################################################################################
# Initialize the ShowManager, position the virtual camera to frame both meshes,
# and launch the rendering loop.
show_m = window.ShowManager(scene=scene, size=(1024, 768), title="FURY Mesh Reader")

camera = show_m.screens[0].camera
camera.local.position = (0.0, -0.15, 5.0)
camera.look_at((0.0, -0.15, 0.0))

show_m.start()
