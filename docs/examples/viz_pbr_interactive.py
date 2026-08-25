"""
====================
Interactive PBR Demo
====================

Physically based rendering (PBR) describes a surface by how it interacts with
light rather than by an ad-hoc colour formula, so the same material reads
convincingly under any lighting. This tutorial puts every property FURY's
``'physical'`` material exposes on a slider, grouped into tabs, so you can watch
each one act on a single sphere in real time.

Most of the light here comes from the environment: a metal has no diffuse
response, so what you see on it is the skybox reflected back. ``Scene(skybox=...)``
hands the cube map to the material as its ``env_map`` automatically. A single
directional light sits on top of that, because **sheen** is retroreflective --
it bounces light back towards wherever it came from, so with environment
lighting alone its characteristic rim barely registers. It is a light in the
scene rather than FURY's default camera-mounted one: a light riding the camera
keeps its highlight pinned to the same spot on the sphere however you orbit,
which on a mirror reads as a headlamp instead of a reflection.

Several properties only mean something on the right kind of surface, which is
what the **Presets** tab is for -- each preset moves the whole material,
including the base colour, to a state where the effect it is named for is the
thing you notice:

* **Index of refraction** and **specular intensity** shape the reflectance of
  *dielectrics*. A metal takes its reflectance from the base colour instead, so
  at ``Metalness = 1`` both are ignored outright and the render does not change
  by a single pixel. Pull Metalness to 0 and they come alive.
* **Sheen** is a cloth layer: a retroreflective rim that reads on a rough
  dielectric, not on polished metal. It is also tinted by ``sheen_color``, which
  defaults to black -- a white tint at full strength simply bleaches the sphere.
  Try the *Velvet* preset.
* **Iridescence** is a thin film whose interference colours are drowned by a
  saturated albedo. It needs a dark, smooth dielectric to show its rainbow
  shift. Try the *Soap bubble* preset.
"""

import numpy as np

from fury.data import fetch_viz_cubemaps, read_viz_cubemap
from fury.io import load_cube_map_texture
from fury.lib import AmbientLight, Buffer, DirectionalLight
from fury.ui import LineSlider2D, TabUI, TextBlock2D, TextButton2D
from fury.window import Scene, ShowManager
import fury.actor as actor

###############################################################################
# Fetch a cube map and build the scene. The skybox is both the backdrop and,
# just as importantly, the light source for the material.

fetch_viz_cubemaps()
cube_map = load_cube_map_texture(read_viz_cubemap("skybox"))

###############################################################################
# The light. ``camera_light=False`` on the ``ShowManager`` below drops the
# default light that rides along with the camera, and this one takes its place,
# pointed from where the skybox's own sun is so that the highlight it casts
# lands on the sun the sphere is already reflecting. Reading that direction off
# the cube map's face layout gets the sign of x wrong; it was measured instead,
# by rendering a mirror sphere from several angles and looking for the light
# direction whose highlight sits on top of the reflected sun -- 30 degrees up,
# and 60 degrees round from +Z towards -X. The sun's glow in this cube map is
# a good ten degrees wide, so there is no more precision to be had than that.
#
# A ``DirectionalLight`` shines from its position towards its target, which is
# the origin by default, so the position is that direction pushed out beyond the
# sphere -- distance does not matter to a directional light, only the direction
# it points. The payoff is a highlight that belongs to the scene: it stays with
# the reflection while the sphere is turned under it, so what the sliders do to
# a highlight can be watched from any angle.

SUN_DIRECTION = np.array([-0.75, 0.50, 0.43])

sun = DirectionalLight(color="#fff4e0", intensity=3.0)
sun.local.position = tuple(40.0 * SUN_DIRECTION)

# Passing ``lights`` replaces the ambient light a bare ``Scene`` would add, so
# put one back: it is what keeps the side facing away from the sun from going
# flat black on the materials the environment map does not reach.
scene = Scene(skybox=cube_map, lights=[AmbientLight(), sun])

###############################################################################
# The sphere itself. ``material='physical'`` selects pygfx's
# ``MeshPhysicalMaterial``, the superset that adds clearcoat, sheen, anisotropy
# and iridescence on top of the metallic-roughness workflow.
#
# Two of the properties below are modulated by a colour that defaults to black,
# which would make their sliders look broken: ``sheen`` is tinted by
# ``sheen_color`` and ``emissive_intensity`` scales ``emissive``. Both colours
# are seeded here so the sliders have something to act on.

INITIAL = {
    "metalness": 1.0,
    "roughness": 0.25,
    "ior": 1.5,
    "specular_intensity": 1.0,
    "env_map": cube_map,
    "env_map_intensity": 1.0,
    "clearcoat": 0.0,
    "clearcoat_roughness": 0.0,
    "anisotropy": 0.0,
    "anisotropy_rotation": 0.0,
    "sheen": 0.0,
    "sheen_roughness": 1.0,
    "iridescence": 0.0,
    "iridescence_ior": 1.3,
    "emissive_intensity": 0.0,
}

###############################################################################
# Four of those properties -- ``clearcoat``, ``sheen``, ``anisotropy`` and
# ``iridescence`` -- are shader gates as much as they are values. pygfx compiles
# each of their branches into the shader only when the property is non-zero at
# the time the material is first drawn; a later write lands in the uniform
# buffer and never triggers a rebuild. Left at 0.0 they are compiled out for
# good, and their sliders -- along with every preset that leans on them -- move
# without changing a pixel. Flooring them at a value far too small to see keeps
# each branch in the shader, ready for the slider to drive.

SHADER_GATED = ("clearcoat", "sheen", "anisotropy", "iridescence")
GATE_EPSILON = 1e-6


def gated(name, value):
    """Keep a shader-gated property non-zero so its branch stays compiled."""
    if name in SHADER_GATED:
        return max(value, GATE_EPSILON)
    return value


sphere_actor = actor.sphere(
    np.zeros((1, 3)),
    colors=(0.97, 0.96, 0.92),
    radii=8.0,
    phi=96,
    theta=96,
    impostor=False,
    material="physical",
    material_params={name: gated(name, value) for name, value in INITIAL.items()},
)
sphere_actor.material.flat_shading = False
###############################################################################
# ``actor.sphere`` bakes the colour into a vertex buffer. Switching the material
# to ``color_mode="auto"`` hands control to ``material.color`` instead, so the
# presets below can restage the sphere -- a metal, a dark bubble, a dyed cloth --
# rather than only nudging its reflectance.

# Silver: near-white reflectance, very slightly warm. On a metal the base
# colour *is* the reflectance, so this is what tints every reflection.
BASE_COLOR = (0.97, 0.96, 0.92, 1.0)
sphere_actor.material.color_mode = "auto"
sphere_actor.material.color = BASE_COLOR
sphere_actor.material.sheen_color = (1.0, 1.0, 1.0)
sphere_actor.material.emissive = (0.35, 0.12, 0.5)

###############################################################################
# ``prim_sphere`` closes the sphere by repeating its first column of vertices at
# longitude +180, so the seam is two coincident rows of *separate* vertices.
# Handed no normals, pygfx averages the face normals meeting at each vertex, and
# each copy of the seam only sees the faces on its own side -- the two disagree
# by half a segment, 5 degrees at the resolution used here. On something mirror
# smooth that reads as the reflection being cut and shoved sideways down one
# meridian, round the back of the default view. A sphere's normals are known
# exactly, so hand them over rather than let them be inferred: centred on the
# origin, the normal is the normalised position.

unit = sphere_actor.geometry.positions.data
unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
sphere_actor.geometry.normals = Buffer(unit.astype(np.float32))

###############################################################################
# Anisotropy stretches the highlight along the surface's tangent direction, and
# something has to supply that direction. With nothing else to go on pygfx
# derives it from the screen-space derivatives of the texture coordinates, which
# are constant across a triangle -- so the highlight breaks into facets -- and
# which jump a whole turn where the longitude wraps, drawing a bright seam down
# the sphere. It also cannot read the 3-component texcoords ``actor.sphere``
# leaves behind, and fails shader validation outright.
#
# A ``tangents`` attribute replaces all of that guesswork. The direction wanted
# is the one along which longitude increases, ``cross(up, n)`` normalised, which
# works out as ``(cos, 0, -sin)`` of the longitude alone. Taking it from the
# longitude rather than from the cross product matters at the poles: there the
# cross product vanishes and has to be replaced by something, and any single
# stand-in direction leaves the fan triangle on the far side interpolating its
# tangent from that stand-in to its opposite -- straight through zero, which
# collapses the tangent frame and lights a speck at each pole on every material.
# The longitude survives at the pole vertices, held in x and z at around 1e-16,
# so each copy still caps its own fan and the tangent stays well defined. The
# fourth component picks the handedness of the bitangent.

lon = np.arctan2(unit[:, 0], unit[:, 2])
tangents = np.column_stack([np.cos(lon), np.zeros_like(lon), -np.sin(lon)])
sphere_actor.geometry.tangents = Buffer(
    np.column_stack([tangents, np.ones(len(tangents))]).astype(np.float32)
)

scene.add(sphere_actor)

material = sphere_actor.material

###############################################################################
# Every control is one row: a label, a slider, and the value the slider prints.
# ``TABS`` is the whole specification -- tab title, then one entry per property
# giving its label, range, starting value and how to apply it -- so adding
# another property is a single line.
#
# ``anisotropy_rotation`` is in radians. The iridescence film slider drives the
# upper bound of ``iridescence_thickness_range`` in nanometres: it is the film
# thickness that decides which colours the interference shifts towards.


def _set(name):
    """Build a callback that writes a slider's value onto the material."""

    def apply(slider):
        setattr(material, name, gated(name, slider.value))

    return apply


def _set_iridescence_thickness(slider):
    material.iridescence_thickness_range = (100.0, slider.value)


#: tab title -> rows of (label, min, max, initial, callback, decimals)
TABS = [
    (
        "Base",
        [
            ("Metalness", 0.0, 1.0, INITIAL["metalness"], _set("metalness"), 2),
            ("Roughness", 0.0, 1.0, INITIAL["roughness"], _set("roughness"), 2),
            ("Index of refraction", 1.0, 2.3, INITIAL["ior"], _set("ior"), 2),
            (
                "Specular intensity",
                0.0,
                1.0,
                INITIAL["specular_intensity"],
                _set("specular_intensity"),
                2,
            ),
            (
                "Environment light",
                0.0,
                3.0,
                INITIAL["env_map_intensity"],
                _set("env_map_intensity"),
                2,
            ),
        ],
    ),
    (
        "Coating",
        [
            ("Clearcoat", 0.0, 1.0, INITIAL["clearcoat"], _set("clearcoat"), 2),
            (
                "Clearcoat roughness",
                0.0,
                1.0,
                INITIAL["clearcoat_roughness"],
                _set("clearcoat_roughness"),
                2,
            ),
            ("Sheen", 0.0, 1.0, INITIAL["sheen"], _set("sheen"), 2),
            (
                "Sheen roughness",
                0.0,
                1.0,
                INITIAL["sheen_roughness"],
                _set("sheen_roughness"),
                2,
            ),
            (
                "Emissive intensity",
                0.0,
                3.0,
                INITIAL["emissive_intensity"],
                _set("emissive_intensity"),
                2,
            ),
        ],
    ),
    (
        "Optical",
        [
            ("Anisotropy", 0.0, 1.0, INITIAL["anisotropy"], _set("anisotropy"), 2),
            (
                "Anisotropy rotation",
                0.0,
                2 * np.pi,
                INITIAL["anisotropy_rotation"],
                _set("anisotropy_rotation"),
                2,
            ),
            (
                "Iridescence",
                0.0,
                1.0,
                INITIAL["iridescence"],
                _set("iridescence"),
                2,
            ),
            (
                "Iridescence IoR",
                1.0,
                2.3,
                INITIAL["iridescence_ior"],
                _set("iridescence_ior"),
                2,
            ),
            (
                "Iridescence film (nm)",
                100.0,
                800.0,
                400.0,
                _set_iridescence_thickness,
                0,
            ),
        ],
    ),
]

###############################################################################
# Lay the rows out in a :class:`fury.ui.TabUI`. Grouping the fifteen properties
# into three tabs keeps the widget short enough to sit beside the sphere, and
# ``TabUI`` is not draggable by default, so it stays where it is put.
#
# Coordinates in ``add_element`` are pixel offsets measured down from the
# content panel's top-left corner. Each slider prints its value just above the
# track, which is why the rows are spaced more generously than the text alone
# would need.

# The tab bar splits its width evenly between the tabs, so each title has to
# fit in TAB_SIZE[0] / n_tabs pixels. Long titles wrap onto a second line and
# spill out of their header into the neighbouring ones.
TAB_SIZE = (420, 260)
ROW_HEIGHT = 44
FIRST_ROW_Y = 34
LABEL_X = 14
SLIDER_X = 195
SLIDER_LENGTH = 120

PRESET_TAB_INDEX = len(TABS)

tab_ui = TabUI(
    position=(20, 20),
    size=TAB_SIZE,
    tab_titles=[title for title, _rows in TABS] + ["Presets"],
    startup_tab_id=0,
    font_size=15,
    draggable=True,
    inactive_color=(0.09, 0.09, 0.12),
)

sliders = []
for tab_index, (_title, rows) in enumerate(TABS):
    for row, (label, low, high, start, callback, decimals) in enumerate(rows):
        y = FIRST_ROW_Y + row * ROW_HEIGHT

        tab_ui.add_element(
            tab_index,
            TextBlock2D(
                text=label,
                font_size=14,
                color=(0.88, 0.88, 0.92),
                vertical_justification="middle",
                dynamic_bbox=True,
            ),
            (LABEL_X, y),
        )

        slider = LineSlider2D(
            initial_value=start,
            min_value=low,
            max_value=high,
            length=SLIDER_LENGTH,
            line_width=5,
            shape="disk",
            outer_radius=7,
            font_size=13,
            text_template=f"{{value:.{decimals}f}}",
        )
        slider.on_moving_slider = callback
        tab_ui.add_element(tab_index, slider, (SLIDER_X, y))
        sliders.append(slider)

###############################################################################
# Presets. Each one restages the whole material so the property it is named for
# is the thing you notice, then pushes the new values back into the sliders so
# the panel keeps telling the truth.

PRESETS = [
    (
        "Polished metal",
        {"color": BASE_COLOR, "metalness": 1.0, "roughness": 0.08},
    ),
    (
        "Brushed metal",
        {
            "color": BASE_COLOR,
            "metalness": 1.0,
            "roughness": 0.45,
            "anisotropy": 1.0,
            "anisotropy_rotation": 1.5,
        },
    ),
    (
        "Car paint",
        {
            "color": (0.55, 0.05, 0.09, 1.0),
            "metalness": 0.9,
            "roughness": 0.5,
            "clearcoat": 1.0,
            "clearcoat_roughness": 0.03,
        },
    ),
    (
        "Velvet",
        {
            "color": (0.25, 0.05, 0.12, 1.0),
            "metalness": 0.0,
            "roughness": 1.0,
            "sheen": 1.0,
            "sheen_roughness": 0.3,
            "sheen_color": (0.9, 0.5, 0.6),
        },
    ),
    (
        "Soap bubble",
        {
            "color": (0.02, 0.02, 0.02, 1.0),
            "metalness": 0.0,
            "roughness": 0.05,
            "iridescence": 1.0,
            "iridescence_ior": 2.0,
            "iridescence_thickness_range": (100.0, 550.0),
        },
    ),
    (
        "Glass",
        {
            "color": (0.85, 0.92, 0.95, 1.0),
            "metalness": 0.0,
            "roughness": 0.02,
            "ior": 1.8,
            "specular_intensity": 1.0,
        },
    ),
]

#: Every property a preset may touch, with the value to fall back to when it
#: does not mention one. Without this a preset would inherit whatever the last
#: one left behind.
PRESET_DEFAULTS = {
    "color": BASE_COLOR,
    "sheen_color": (1.0, 1.0, 1.0),
    "iridescence_thickness_range": (100.0, 400.0),
    **INITIAL,
}

#: slider label -> the material property it drives, for syncing after a preset
SLIDER_PROPERTY = {
    "Metalness": "metalness",
    "Roughness": "roughness",
    "Index of refraction": "ior",
    "Specular intensity": "specular_intensity",
    "Environment light": "env_map_intensity",
    "Clearcoat": "clearcoat",
    "Clearcoat roughness": "clearcoat_roughness",
    "Sheen": "sheen",
    "Sheen roughness": "sheen_roughness",
    "Emissive intensity": "emissive_intensity",
    "Anisotropy": "anisotropy",
    "Anisotropy rotation": "anisotropy_rotation",
    "Iridescence": "iridescence",
    "Iridescence IoR": "iridescence_ior",
}

SLIDER_ROWS = [row for _title, rows in TABS for row in rows]


def apply_preset(values):
    """Put the material into a named state and resync every slider."""
    settings = dict(PRESET_DEFAULTS)
    settings.update(values)
    for name, value in settings.items():
        setattr(material, name, gated(name, value))

    for slider, (label, *_rest) in zip(sliders, SLIDER_ROWS, strict=True):
        if label.startswith("Iridescence film"):
            slider.value = settings["iridescence_thickness_range"][1]
        else:
            slider.value = settings[SLIDER_PROPERTY[label]]


###############################################################################
# The buttons themselves. ``TextButton2D`` takes the same ``states`` mapping
# viz_button.py demonstrates -- one entry per state the button can resolve --
# and here each state is a colour rather than a colour and a caption, so the
# label stays put and only the plate under it reacts: it lifts on hover, and the
# preset currently on the sphere stays in the accent colour. That last part is
# what ``is_toggle`` buys; the click handler switches the others off, which
# turns six independent buttons into one group with a current selection.

PRESET_STATES = {
    "default": (0.20, 0.21, 0.27),
    "hover": (0.28, 0.30, 0.40),
    "pressed": (0.35, 0.44, 0.68),
    "disabled": (0.12, 0.12, 0.15),
}
PRESET_LABEL_COLOR = (0.90, 0.91, 0.96)

# Two columns of three, sized from the panel rather than by eye: the grid is
# centred in TAB_SIZE[0] and starts below the tab bar, so every button lands
# inside the content area with room to spare.
BUTTON_SIZE = (186, 38)
BUTTON_GAP = (12, 12)
BUTTON_COLUMNS = 2
GRID_WIDTH = BUTTON_COLUMNS * BUTTON_SIZE[0] + (BUTTON_COLUMNS - 1) * BUTTON_GAP[0]
GRID_X = (TAB_SIZE[0] - GRID_WIDTH) // 2
GRID_Y = FIRST_ROW_Y  # start on the same line the slider tabs start on

preset_buttons = []
preset_status = TextBlock2D(
    text="Pick a preset",
    font_size=13,
    vertical_justification="middle",
    dynamic_bbox=True,
)


def _select_preset(name, values):
    """Apply a preset and leave its own button lit, the others dark."""

    def clicked(button):
        apply_preset(values)
        for other in preset_buttons:
            other.toggled = other is button
        preset_status.message = f"On the sphere: {name}"

    return clicked


for index, (name, values) in enumerate(PRESETS):
    row, column = divmod(index, BUTTON_COLUMNS)
    button = TextButton2D(
        label=name,
        states=PRESET_STATES,
        size=BUTTON_SIZE,
        font_size=14,
        is_toggle=True,
    )
    button.child.color = PRESET_LABEL_COLOR
    button.on_clicked = _select_preset(name, values)
    tab_ui.add_element(
        PRESET_TAB_INDEX,
        button,
        (
            GRID_X + column * (BUTTON_SIZE[0] + BUTTON_GAP[0]),
            GRID_Y + row * (BUTTON_SIZE[1] + BUTTON_GAP[1]),
        ),
    )
    preset_buttons.append(button)

tab_ui.add_element(
    PRESET_TAB_INDEX,
    preset_status,
    (GRID_X, GRID_Y + 3 * (BUTTON_SIZE[1] + BUTTON_GAP[1]) + 6),
)

scene.add(tab_ui)

###############################################################################
# Frame the sphere with ``camera.show_object``. That does more than place the
# camera: it also leaves the controller orbiting about the sphere, so dragging
# rotates around the object rather than about some unrelated point. ``scale``
# pulls the camera back a little so the sphere clears the tab widget.
#
# The default camera light stays on. Turning it off leaves the skybox as the
# only illumination, which looks fine for metals but flattens sheen to almost
# nothing -- a retroreflective lobe needs a light near the viewer to show up.

show_manager = ShowManager(
    scene=scene,
    size=(1360, 820),
    title="FURY Interactive PBR Demo",
    camera_light=False,
)

camera = show_manager.screens[0].camera
camera.show_object(sphere_actor, scale=1.6)

###############################################################################
# Keep the tab widget anchored to the top-left corner as the window is resized.


def keep_tabs_anchored(size):
    tab_ui.set_position((20, 20))


show_manager.resize_callback(keep_tabs_anchored)

if __name__ == "__main__":
    show_manager.start()
