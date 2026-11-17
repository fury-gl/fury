"""
===========================================
Physically-Based Rendering (PBR) on spheres
===========================================

Physically based rendering describes a surface by how it interacts with light,
so one material reads convincingly under any lighting. This tutorial lays the
main PBR properties of FURY's ``'physical'`` material out side by side: every
row sweeps one property from its lowest to its highest value across eleven
spheres, and everything else stays fixed.

:func:`fury.actor.manifest_pbr` makes that a single call per row. Given an actor
factory such as :func:`fury.actor.sphere` and one center per object, it builds
one mesh per center and gives each of them its own PBR material.
"""

import numpy as np

from fury.data import fetch_viz_cubemaps, read_viz_cubemap
from fury.io import load_cube_map_texture
from fury.lib import AmbientLight, DirectionalLight
from fury.window import Scene, ShowManager
import fury.actor as actor

###############################################################################
# Most of the light comes from the environment: a metal has no diffuse
# response, so what you see on it is the skybox reflected back. A directional
# light is added on top, placed where the skybox's own sun is, so its highlight
# lands on the sun the spheres are already reflecting.

fetch_viz_cubemaps()
cube_map = load_cube_map_texture(read_viz_cubemap("skybox"))

SUN_DIRECTION = np.array([-0.75, 0.50, 0.43])

sun = DirectionalLight(color="#fff4e0", intensity=3.0)
sun.local.position = tuple(40.0 * SUN_DIRECTION)

scene = Scene(skybox=cube_map, lights=[AmbientLight(), sun])

###############################################################################
# Each row is ``(label, property, values, color, fixed)``. ``property`` is the
# one swept over ``values`` and ``fixed`` holds the properties that stay the
# same along the row, so the swept effect is the only thing that changes.
#
# Some properties only show on the right surface: anisotropy stretches the
# highlight of a metal, clearcoat adds a glossy layer over a rough base, and the
# index of refraction shapes the reflectance of a dielectric, which is why that
# row is not metallic.

NUM_VALUES = 11
UNIT = np.linspace(0, 1, NUM_VALUES)

ROWS = [
    ("Metalness", "metalness", UNIT, (1.0, 0.85, 0.3), {"roughness": 0.2}),
    ("Roughness", "roughness", UNIT, (0.2, 0.4, 1.0), {"metalness": 1.0}),
    (
        "Anisotropy",
        "anisotropy",
        UNIT,
        (0.9, 0.9, 0.9),
        {"metalness": 1.0, "roughness": 0.5},
    ),
    (
        "Anisotropy rotation (0 to pi)",
        "anisotropy_rotation",
        np.linspace(0, np.pi, NUM_VALUES),
        (0.9, 0.9, 0.9),
        {"anisotropy": 1.0, "metalness": 1.0, "roughness": 0.5},
    ),
    ("Clearcoat", "clearcoat", UNIT, (0.1, 0.7, 0.7), {"roughness": 0.6}),
    (
        "Clearcoat roughness",
        "clearcoat_roughness",
        UNIT,
        (0.1, 0.7, 0.7),
        {"clearcoat": 1.0, "roughness": 0.6},
    ),
    (
        "Index of refraction (1.0 to 2.3)",
        "ior",
        np.linspace(1.0, 2.3, NUM_VALUES),
        (0.1, 0.7, 0.7),
        {"roughness": 0.2},
    ),
]

###############################################################################
# One call builds a whole row. The named PBR arguments in ``fixed`` are shared
# by all eleven spheres, while ``material_params`` is a list holding one dict
# per center, so each sphere gets its own value of the swept property. Any
# ``PBRMaterial`` property is accepted there, for instance ``sheen`` or
# ``iridescence``. The remaining keyword arguments (``colors``, ``radii``, ...)
# go to :func:`fury.actor.sphere`.
#
# The group is added to the scene after the call, which lets the scene hand the
# skybox to every new material as its environment map.

spheres_group = actor.Group()

for i, (_, prop, values, color, fixed) in enumerate(ROWS):
    centers = np.column_stack(
        (
            -25 + 5 * np.arange(NUM_VALUES),
            np.full(NUM_VALUES, -5.0 * i),
            np.zeros(NUM_VALUES),
        )
    )
    spheres = actor.manifest_pbr(
        actor.sphere,
        centers=centers,
        material_params=[{prop: value} for value in values],
        colors=color,
        radii=2,
        phi=32,
        theta=32,
        **fixed,
    )
    spheres_group.add(spheres)

scene.add(spheres_group)

###############################################################################
# Labels: the row names on the left and the swept value above the first row.
# The columns use a 0 to 1 scale; the two rows with another range name it in
# their label.

row_labels = actor.text(
    [label for label, *_ in ROWS],
    position=[(-29, -5 * i, 0) for i in range(len(ROWS))],
    font_size=1.2,
    anchor="middle-right",
    outline_thickness=0.2,
)
column_labels = actor.text(
    [f"{value:.1f}" for value in UNIT],
    position=[(-25 + 5 * j, 3.5, 0) for j in range(NUM_VALUES)],
    font_size=1.0,
    outline_thickness=0.2,
)
scene.add(row_labels)
scene.add(column_labels)

show_manager = ShowManager(
    scene=scene,
    size=(1200, 900),
    title="FURY PBR Spheres",
)

show_manager.screens[0].camera.show_object(spheres_group, scale=1.1)

###############################################################################
# ``manifest_pbr`` also converts a mesh that already exists, which is a second
# window so the grid above stays a pure sweep. Two identical spheres are built
# with the default phong material and the right one is switched to a glossy PBR
# material in place, keeping its color. Passing a mesh instead of a factory takes
# no centers, and the call returns the new material. Convert before adding the
# sphere to the scene, so the scene can hand the skybox to the new material.

phong_sphere = actor.sphere(
    np.array([[-3.0, 0.0, 0.0]]),
    colors=(0.8, 0.2, 0.2),
    radii=2,
    phi=48,
    theta=48,
    impostor=False,
)
pbr_sphere = actor.sphere(
    np.array([[3.0, 0.0, 0.0]]),
    colors=(0.8, 0.2, 0.2),
    radii=2,
    phi=48,
    theta=48,
    impostor=False,
)
actor.manifest_pbr(pbr_sphere, metalness=0.0, roughness=0.3, clearcoat=1.0)

convert_labels = actor.text(
    ["Phong", "PBR, converted in place"],
    position=[(-3, -3.5, 0), (3, -3.5, 0)],
    font_size=0.8,
    anchor="top-center",
    outline_thickness=0.2,
)

convert_sun = DirectionalLight(color="#fff4e0", intensity=3.0)
convert_sun.local.position = tuple(40.0 * SUN_DIRECTION)
convert_scene = Scene(skybox=cube_map, lights=[AmbientLight(), convert_sun])
convert_group = actor.Group()
convert_group.add(phong_sphere, pbr_sphere, convert_labels)
convert_scene.add(convert_group)

convert_manager = ShowManager(
    scene=convert_scene,
    size=(900, 600),
    title="FURY PBR Conversion",
)
convert_manager.screens[0].camera.show_object(convert_group, scale=1.1)

if __name__ == "__main__":
    show_manager.start()
    convert_manager.start()
