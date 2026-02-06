"""
Streamtube Colors
=================

Demonstrates streamtube color modes: single color, per-line colors,
and direction-based RGB coloring computed on the GPU.
"""

import numpy as np

import fury

scene = fury.window.Scene()
scene.background = (0.1, 0.1, 0.1)


def make_helix(center, axis, n_points=100, radius=2.0, pitch=0.5, turns=3):
    t = np.linspace(0, turns * 2 * np.pi, n_points)
    if axis == "x":
        pts = np.column_stack([
            center[0] + pitch * t,
            center[1] + radius * np.cos(t),
            center[2] + radius * np.sin(t),
        ])
    elif axis == "y":
        pts = np.column_stack([
            center[0] + radius * np.cos(t),
            center[1] + pitch * t,
            center[2] + radius * np.sin(t),
        ])
    else:
        pts = np.column_stack([
            center[0] + radius * np.cos(t),
            center[1] + radius * np.sin(t),
            center[2] + pitch * t,
        ])
    return pts.astype(np.float32)


def make_wave(start, end, n_points=80, amplitude=1.5, freq=3):
    t = np.linspace(0, 1, n_points)
    direction = np.array(end) - np.array(start)
    length = np.linalg.norm(direction)
    d = direction / length
    perp = np.cross(d, [0, 1, 0])
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross(d, [1, 0, 0])
    perp = perp / np.linalg.norm(perp)
    pts = (
        np.array(start)[None, :]
        + np.outer(t, direction)
        + amplitude * np.sin(freq * 2 * np.pi * t)[:, None] * perp[None, :]
    )
    return pts.astype(np.float32)


lines_single = [
    make_helix([-15, 0, 0], "x"),
    make_helix([-15, 0, 4], "x", radius=1.5, pitch=0.6),
    make_helix([-15, 0, -4], "x", radius=1.0, pitch=0.4),
    make_wave([-15, 4, 2], [-5, 4, 2]),
    make_wave([-15, -4, -2], [-5, -4, -2], amplitude=2.0),
]

tube_single = fury.actor.streamtube(
    lines_single,
    colors=(0.2, 0.7, 1.0),
    radius=0.15,
    segments=8,
)
scene.add(tube_single)

lines_perline = [
    make_helix([0, 0, 0], "y"),
    make_helix([4, 0, 0], "y", radius=1.5, pitch=0.6),
    make_helix([-4, 0, 0], "y", radius=1.0, pitch=0.4),
    make_wave([2, -2, 3], [2, 12, 3]),
    make_wave([-2, -2, -3], [-2, 12, -3], amplitude=2.0),
]

per_line_colors = np.array([
    [1.0, 0.2, 0.2],
    [0.2, 1.0, 0.2],
    [0.2, 0.2, 1.0],
    [1.0, 1.0, 0.2],
    [1.0, 0.2, 1.0],
], dtype=np.float32)

tube_perline = fury.actor.streamtube(
    lines_perline,
    colors=per_line_colors,
    radius=0.15,
    segments=8,
)
scene.add(tube_perline)

lines_rgb = [
    make_helix([15, 0, 0], "z"),
    make_helix([19, 0, 0], "z", radius=1.5, pitch=0.6),
    make_helix([11, 0, 0], "z", radius=1.0, pitch=0.4),
    make_wave([13, 2, -2], [13, 2, 12]),
    make_wave([17, -2, -2], [17, -2, 12], amplitude=2.0),
]

tube_rgb = fury.actor.streamtube(
    lines_rgb,
    colors="rgb",
    radius=0.15,
    segments=8,
)
scene.add(tube_rgb)

n = 50
length = 10.0
origin = np.array([0, -15, 0], dtype=np.float32)

x_line = np.column_stack([
    np.linspace(0, length, n),
    np.zeros(n),
    np.zeros(n),
]).astype(np.float32) + origin

y_line = np.column_stack([
    np.zeros(n),
    np.linspace(0, length, n),
    np.zeros(n),
]).astype(np.float32) + origin

z_line = np.column_stack([
    np.zeros(n),
    np.zeros(n),
    np.linspace(0, length, n),
]).astype(np.float32) + origin

diag_xy = np.column_stack([
    np.linspace(0, length, n),
    np.linspace(0, length, n),
    np.zeros(n),
]).astype(np.float32) + origin

diag_xz = np.column_stack([
    np.linspace(0, length, n),
    np.zeros(n),
    np.linspace(0, length, n),
]).astype(np.float32) + origin

diag_yz = np.column_stack([
    np.zeros(n),
    np.linspace(0, length, n),
    np.linspace(0, length, n),
]).astype(np.float32) + origin

diag_xyz = np.column_stack([
    np.linspace(0, length, n),
    np.linspace(0, length, n),
    np.linspace(0, length, n),
]).astype(np.float32) + origin

axes_rgb = fury.actor.streamtube(
    [x_line, y_line, z_line, diag_xy, diag_xz, diag_yz, diag_xyz],
    colors="rgb",
    radius=0.2,
    segments=8,
)
scene.add(axes_rgb)
print(tube_perline.geometry.colors.data)

showm = fury.window.ShowManager(scene=scene, size=(1400, 700))
showm.start()
