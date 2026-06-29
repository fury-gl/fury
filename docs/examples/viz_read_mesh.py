"""
=================================
Reading Meshes, Lines, and Points
=================================

This example demonstrates how to fetch 3D data assets from the
``polyxios-data`` repository, read them into NumPy arrays using FURY's
dedicated I/O methods, and render them as distinct colored actors side by side:
a surface mesh, a collection of lines, and a point cloud.

Polyxios is the lightweight, dependency-free I/O backend used by FURY. It
is designed to auto-detect file formats by extension and map complex scientific
surface structures (``.vtk``, ``.vtp``, ``.ply``, ``.obj``, ``.mesh``)
seamlessly into memory-contiguous NumPy arrays, and its ``fetch`` helper downloads
and caches sample assets so tutorials and tests can run without bundling large files.

This example highlights three fundamental data archetypes:

* **Surface Meshes (via ``read_mesh``):** Returns a triple of
  ``(vertices, faces, colors)`` where ``vertices`` is an ``(N, 3)`` float32 array,
  ``faces`` is an ``(M, 3)`` int32 array of triangulated indices, and ``colors`` is an
  optional ``(N, 3)`` float32 color map.

* **Line Streams (via ``read_lines``):** Returns a tuple of ``(lines, colors)``. Here,
  ``lines`` is a list of independent ``(P, 3)`` float32 arrays (where each array
  represents an individual continuous stroke or fiber pathway), and ``colors`` contains
  a corresponding list of per-vertex color maps.

* **Point Clouds (via ``read_points``):** Returns a tuple of ``(points, colors)`` where
  ``points`` is an ``(N, 3)`` float32 coordinate block representing un-connected point
  data, and ``colors`` is an optional per-point attribute array.
"""

#########################################################################################
# Import the required libraries.
import numpy as np
import polyxios as px

from fury import actor, ui, window
from fury.io import read_lines, read_mesh, read_points

#########################################################################################
# Define helper to process geometry, normalize space, and apply fallback gradients.


def points_to_actor(file_path):
    """Read a point cloud file and build a colored point actor."""
    points, colors = read_points(file_path)

    if points.size == 0:
        raise ValueError(f"No points found in file: {file_path}")

    # Center on the origin and normalize the scale to roughly unit size
    points = points - points.mean(axis=0)
    points = (points / np.max(np.abs(points))).astype(np.float32)

    # Use the file's colors when present, otherwise apply a height-based gradient
    if colors is None:
        height = points[:, 1]
        height_range = height.max() - height.min()
        y_range = height_range if height_range > 0 else 1.0

        t = ((height - height.min()) / y_range)[:, None]
        low_color = np.array([0.10, 0.20, 0.70], dtype=np.float32)
        high_color = np.array([0.95, 0.75, 0.20], dtype=np.float32)
        colors = (low_color * (1.0 - t) + high_color * t).astype(np.float32)

    # ``actor.point`` turns an (N, 3) array of point positions and their
    # matching per-point colors into an optimized graphic object.
    point_actor = actor.point(points, colors=colors)
    return point_actor, len(points)


def lines_to_actor(file_path):
    """Read a lines file and build a colored stream/line actor."""
    lines, colors = read_lines(file_path)

    if not lines:
        raise ValueError(f"No line elements found in file: {file_path}")

    # Stack all points temporarily to calculate overall centering and scaling metrics
    all_points = np.vstack(lines)
    center = all_points.mean(axis=0)
    max_scale = np.max(np.abs(all_points - center))

    # Center on the origin and normalize the scale to roughly unit size
    normalized_lines = [
        ((line - center) / max_scale).astype(np.float32) for line in lines
    ]

    # Use the file's colors when present, otherwise apply a height-based gradient
    if colors is None:
        # Re-stack lines to compute a global bounding box for the height gradient
        all_norm_points = np.vstack(normalized_lines)
        y_min = all_norm_points[:, 1].min()
        y_max = all_norm_points[:, 1].max()
        y_range = y_max - y_min if (y_max - y_min) > 0 else 1.0

        low_color = np.array([0.10, 0.20, 0.70], dtype=np.float32)
        high_color = np.array([0.95, 0.75, 0.20], dtype=np.float32)

        colors = []
        for line in normalized_lines:
            heights = line[:, 1]
            t = ((heights - y_min) / y_range)[:, None]
            line_colors = (low_color * (1.0 - t) + high_color * t).astype(np.float32)
            colors.append(line_colors)

    # ``actor.streamlines`` accepts a list of coordinate arrays and color arrays
    line_actor = actor.streamlines(normalized_lines, colors=colors)

    # Calculate totals for reporting
    total_vertices = sum(len(line) for line in lines)
    total_lines = len(lines)

    return line_actor, total_vertices, total_lines


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
# Fetch the sample data assets via polyxios.
#
# * ``Human.vtp`` a surface mesh possessing native topological faces and color tracks.
# * ``hello.vtk`` contains line elements forming spatial lettering out of poly-line.
# * ``star.mesh`` handles plain point coordinates lacking explicitly structured links.
human_path = px.fetch("Human.vtp")
line_path = px.fetch("hello.vtk")
ball_path = px.fetch("star.mesh")

# Generate the actors and fetch metadata counts
mesh_actor, mesh_nv, mesh_nf = mesh_to_actor(human_path)
line_actor, line_nv, line_nl = lines_to_actor(line_path)
point_actor, point_nv = points_to_actor(ball_path)

print(f"Human.vtp (Mesh): {mesh_nv} vertices, {mesh_nf} faces")
print(f"hello.vtk (Lines): {line_nv} vertices across {line_nl} lines")
print(f"star.mesh (Points): {point_nv} points")

#########################################################################################
# Coordinate alignment transforms.
# ``Human.vtp`` is modeled lying along its Z axis, so by default it faces the
# camera end-on. Every FURY actor exposes transform helpers (``rotate``,
# ``translate``, ``scale``), so we stand it upright with a -90 degrees rotation
# about the X axis, mapping its head-to-toe axis to the vertical and apply 180 degrees
# so it faces the camera.
mesh_actor.rotate((-90, 180, 0))

#########################################################################################
# Place the three actors side by side (Left, Center, Right) to avoid overlapping.
mesh_actor.local.position = (-3.0, 0.0, 0.0)
line_actor.local.position = (0.0, 0.0, 0.0)
point_actor.local.position = (3.0, 0.0, 0.0)

#########################################################################################
# Add matching 3D text labels directly beneath each of the three objects.
labels_actor = actor.text(
    ["Mesh (Human.vtp)", "Lines (hello.vtk)", "Points (star.mesh)"],
    position=[(-3.0, -1.25, 0.0), (0.0, -1.25, 0.0), (3.0, -1.25, 0.0)],
    colors=(0.9, 0.9, 0.95),
    font_size=0.16,
    anchor="top-center",
)

#########################################################################################
# Set up the 3D scene and register all visual elements.
scene = window.Scene(background=(0.05, 0.05, 0.08))
scene.add(mesh_actor)
scene.add(line_actor)
scene.add(point_actor)
scene.add(labels_actor)

#########################################################################################
# Add a 2D text HUD overlay detailing what each object represents.
info_text = (
    f"FURY x Polyxios Geometry Reader\n"
    f"Left:   Human.vtp (Mesh: {mesh_nv} verts, {mesh_nf} faces)\n"
    f"Center: hello.vtk (Lines: {line_nv} total vertices, {line_nl} lines)\n"
    f"Right:  star.mesh (Points: {point_nv} coordinates)\n"
    f"Decoupled data paths processed into native NumPy arrays."
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
# Initialize the ShowManager, frame the camera to fit the wider 3-element layout,
# and initialize the rendering window loop.
show_m = window.ShowManager(
    scene=scene, size=(1280, 728), title="FURY Multimodal Reader"
)

camera = show_m.screens[0].camera
camera.local.position = (0.0, -0.15, 6.0)
camera.look_at((0.0, -0.15, 0.0))

show_m.start()
