"""
====================================
Jungle Ecosystem Survival Simulation
====================================

An interactive 3D jungle ecosystem survival simulation containing Lions, Elephants,
and Deers navigating land, lakes, rivers, and vegetation patches.
Features survival states (thirst, hunger, aging, mating, hunting, and death),
a dynamic minimap, WASD free fly camera, and diagnostic control interfaces.
"""

import numpy as np
import pygfx as gfx
from fury import actor, ui, window
from fury.window import EventType

# Simulation dimensions and constants
JUNGLE_SIZE = 600.0
LAKE_RADIUS = 55.0
LAKE_CENTER = np.array([0.0, 0.0, 0.0])

# Species characteristics
MAX_HEALTH = {"lion": 150.0, "elephant": 500.0, "deer": 100.0}
MAX_AGE = {"lion": 90.0, "elephant": 140.0, "deer": 75.0}
BASE_SPEED = {"lion": 7.0, "elephant": 3.5, "deer": 10.0}
RUN_SPEED = {"lion": 15.5, "elephant": 8.5, "deer": 14.0}
MIN_SPEED = 2.0
ANIMAL_COLOR = {
    "lion": (0.9, 0.45, 0.1),
    "elephant": (0.45, 0.45, 0.5),
    "deer": (0.55, 0.35, 0.2),
}

# Grouping / Flocking coefficients per species
GROUP_COEFFS = {
    "lion": {"cohesion": 1.0, "alignment": 0.8, "separation": 1.2},
    "elephant": {"cohesion": 2.2, "alignment": 0.2, "separation": 1.8},
    "deer": {"cohesion": 1.8, "alignment": 1.2, "separation": 1.5},
}

# Vegetation patch centers
veg_patches = [
    np.array([-150.0, 0.0, -150.0]),
    np.array([160.0, 0.0, 160.0]),
    np.array([-180.0, 0.0, 120.0]),
    np.array([180.0, 0.0, -150.0]),
    np.array([0.0, 0.0, -220.0]),
]


# Global simulation state
state = {
    "selected_animal": None,
    "screen_size": (1024.0, 768.0),
    "animals": [],
    "next_animal_id": 0,
    "minimap_dots_actor": None,
    "minimap_selected_actor": None,
    # Camera states
    "keys": set(),
    "cam_yaw": 0.0,
    "cam_pitch": -np.radians(45.0),
    "is_dragging_cam": False,
    "last_mouse": None,
    # Simulation factors and rates
    "sim_speed": 1.0,
    "cohesion_mult": 1.0,
    "separation_mult": 1.0,
    # Global multipliers
    "global_speed_factor": 1.0,
    "global_hunger_factor": 1.0,
    "global_thirst_factor": 1.0,
    "global_age_factor": 1.0,
    "global_mating_factor": 1.0,
    "global_hunting_factor": 1.0,
    # Species factors
    "lion_speed_factor": 1.0,
    "lion_hunger_factor": 1.0,
    "lion_thirst_factor": 1.0,
    "lion_age_factor": 1.0,
    "lion_mating_factor": 1.0,
    "lion_hunting_factor": 1.0,
    "elephant_speed_factor": 1.0,
    "elephant_hunger_factor": 1.0,
    "elephant_thirst_factor": 1.0,
    "elephant_age_factor": 1.0,
    "elephant_mating_factor": 1.0,
    "elephant_hunting_factor": 1.0,
    "deer_speed_factor": 1.0,
    "deer_hunger_factor": 1.0,
    "deer_thirst_factor": 1.0,
    "deer_age_factor": 1.0,
    "deer_mating_factor": 1.0,
    "deer_hunting_factor": 1.0,
}

# Setup Scene
scene = window.Scene()
scene.background = (0.5, 0.7, 0.95)


# Height map generator for rolling hills and valleys terrain
def get_terrain_height(x, z):
    # Base height function (sine/cosine waves)
    y = 12.0 * np.sin(x / 45.0) * np.cos(z / 45.0) + 6.0 * np.cos(x / 90.0)

    # Flatten lake
    d_lake = np.sqrt(x**2 + z**2)
    lake_factor = np.clip((d_lake - LAKE_RADIUS) / 25.0, 0.0, 1.0)
    return y * lake_factor


# Procedural Terrain Generation
def generate_terrain(grid_size=60, size=600.0):
    M = grid_size + 1
    N = grid_size + 1
    xs = np.linspace(-size / 2, size / 2, M)
    zs = np.linspace(-size / 2, size / 2, N)

    vertices = []
    colors = []

    for r in range(M):
        for c in range(N):
            x = xs[r]
            z = zs[c]
            y = get_terrain_height(x, z)
            vertices.append([x, y, z])

            # Color mapping by elevation
            h_norm = np.clip((y + 15.0) / 30.0, 0.0, 1.0)
            col = (
                0.05 + 0.1 * h_norm,
                0.12 + 0.08 * h_norm,
                0.05 + 0.05 * h_norm,
                1.0,
            )
            colors.append(col)

    vertices = np.array(vertices, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)

    faces = []
    for r in range(M - 1):
        for c in range(N - 1):
            idx0 = r * N + c
            idx1 = (r + 1) * N + c
            idx2 = r * N + (c + 1)
            idx3 = (r + 1) * N + (c + 1)

            faces.append([idx0, idx1, idx2])
            faces.append([idx1, idx3, idx2])

    faces = np.array(faces, dtype=np.int32)
    return vertices, faces, colors


# Generate Ground surface
verts, faces, cols = generate_terrain()
ground = actor.surface(verts, faces=faces, colors=cols)
scene.add(ground)

# Lake surface
lake = actor.cylinder(
    centers=np.array([[LAKE_CENTER[0], 0.02, LAKE_CENTER[2]]]),
    directions=np.array([[0.0, 1.0, 0.0]]),
    colors=(0.1, 0.4, 0.75),
    height=0.08,
    radii=LAKE_RADIUS,
)
scene.add(lake)


# Vegetation patches (represented by small green cones on terrain)
for vp in veg_patches:
    for _ in range(12):
        offset_x = np.random.uniform(-15.0, 15.0)
        offset_z = np.random.uniform(-15.0, 15.0)
        wx = vp[0] + offset_x
        wz = vp[2] + offset_z
        wy = get_terrain_height(wx, wz) + 0.5
        c_pos = np.array([wx, wy, wz])
        sh = np.random.uniform(1.0, 2.0)
        bush = actor.cone(
            centers=np.array([c_pos]),
            directions=np.array([[0.0, 1.0, 0.0]]),
            colors=(0.15, 0.5, 0.2),
            height=sh,
            radii=sh * 0.4,
        )
        scene.add(bush)

# Lights
ambient = gfx.AmbientLight(color=(0.55, 0.6, 0.55), intensity=1.8)
scene.add(ambient)


# Quaternion helpers
def axis_angle_to_quat(axis, angle_deg):
    angle_rad = np.radians(angle_deg)
    s = np.sin(angle_rad / 2.0)
    c = np.cos(angle_rad / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c])


def quat_mult(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def rotate_vector(quat, vec):
    q_vec = quat[:3]
    q_w = quat[3]
    uv = np.cross(q_vec, vec)
    uv_cross = np.cross(q_vec, uv)
    return vec + 2.0 * (q_w * uv + uv_cross)


def disable_depth_testing(world_object):
    if hasattr(world_object, "material") and world_object.material is not None:
        world_object.material.depth_test = False
        world_object.material.depth_write = False
    if hasattr(world_object, "children"):
        for child in world_object.children:
            disable_depth_testing(child)


# Animal Spawn Helper
def spawn_animal(species, position, is_child=False):
    color = ANIMAL_COLOR[species]
    h_offset = 0.6 if species == "lion" else 0.5
    if species == "elephant":
        h_offset = 1.0

    position[1] = get_terrain_height(position[0], position[2]) + h_offset

    if species == "lion":
        art = actor.ellipsoid(
            centers=np.array([[0.0, 0.0, 0.0]]),
            lengths=(2.0, 1.0, 1.0),
            colors=color,
        )
    elif species == "elephant":
        art = actor.box(
            centers=np.array([[0.0, 0.0, 0.0]]), colors=color, scales=(3.2, 1.8, 1.6)
        )
    else:
        art = actor.cone(
            centers=np.array([[0.0, 0.0, 0.0]]),
            directions=np.array([[0.0, 0.0, 1.0]]),
            colors=color,
            height=1.5,
            radii=0.45,
        )

    art.local.position = position
    scene.add(art)

    animal = {
        "id": state["next_animal_id"],
        "species": species,
        "pos": np.array(position, dtype=np.float32),
        "vel": np.random.randn(3).astype(np.float32),
        "actor": art,
        "age": np.random.uniform(16.0, 45.0) if not is_child else 0.0,
        "health": MAX_HEALTH[species],
        "hunger": np.random.uniform(10.0, 30.0),
        "thirst": np.random.uniform(10.0, 30.0),
        "is_child": is_child,
        "gender": np.random.choice(["M", "F"]),
        "cooldown": np.random.uniform(5.0, 15.0),
        "hunt_target_id": None,
    }
    animal["vel"][1] = 0.0
    speed = BASE_SPEED[species]
    animal["vel"] = (animal["vel"] / (np.linalg.norm(animal["vel"]) + 1e-5)) * speed

    if is_child:
        art.local.scale = [0.4, 0.4, 0.4]

    art.agent_idx = animal["id"]
    state["animals"].append(animal)
    state["next_animal_id"] += 1
    return animal


# Initial Population Spawning
for _ in range(55):
    rx = np.random.uniform(-250.0, 250.0)
    rz = np.random.uniform(-250.0, 250.0)
    spawn_animal("deer", np.array([rx, 0.0, rz]))

for _ in range(12):
    rx = np.random.uniform(-250.0, 250.0)
    rz = np.random.uniform(-250.0, 250.0)
    spawn_animal("lion", np.array([rx, 0.0, rz]))

for _ in range(10):
    rx = np.random.uniform(-250.0, 250.0)
    rz = np.random.uniform(-250.0, 250.0)
    spawn_animal("elephant", np.array([rx, 0.0, rz]))


# Project 3D to Screen Coordinates
def world_to_screen(world_pos, camera, screen_size):
    p4 = np.array([world_pos[0], world_pos[1], world_pos[2], 1.0])
    v = camera.view_matrix
    p = camera.projection_matrix
    clip = p @ (v @ p4)
    if clip[3] == 0:
        return np.array([0.0, 0.0])
    ndc = clip[:3] / clip[3]
    w, h = screen_size
    screen_x = (ndc[0] + 1.0) * 0.5 * w
    screen_y = (1.0 - ndc[1]) * 0.5 * h
    return np.array([screen_x, screen_y])


# UI Redesign - Info Panel (Height expanded to 280 to hold two-column status)
info_panel = ui.Panel2D(
    size=(320, 280), color=(0.06, 0.09, 0.06), has_border=True, border_width=2
)
info_panel.set_position((15, 15))

lbl_title = ui.TextBlock2D(
    text="JUNGLE ECOSYSTEM",
    position=(20, 15),
    font_size=16,
    color=(0.9, 0.75, 0.2),
    bold=True,
    dynamic_bbox=True,
)
info_panel.add_element(lbl_title, (20, 15))

lbl_legend = ui.TextBlock2D(
    text="Lions: 0 | Elephants: 0 | Deers: 0",
    position=(20, 42),
    font_size=13,
    color=(0.85, 0.85, 0.85),
    dynamic_bbox=True,
)
info_panel.add_element(lbl_legend, (20, 42))

# Snug fit Animal focus buttons placed inside Info panel below legend
btn_states_lion = {
    "hover": {"text": "LION", "color": (0.3, 0.4, 0.3)},
    "pressed": {"text": "LION", "color": (0.1, 0.2, 0.1)},
    "default": {"text": "LION", "color": (0.15, 0.25, 0.15)},
}
btn_lion = ui.TextButton2D(
    label="LION",
    size=(80, 26),
    position=(20, 68),
    font_size=14,
    states=btn_states_lion,
)

btn_states_ele = {
    "hover": {"text": "ELEPHANT", "color": (0.3, 0.4, 0.3)},
    "pressed": {"text": "ELEPHANT", "color": (0.1, 0.2, 0.1)},
    "default": {"text": "ELEPHANT", "color": (0.15, 0.25, 0.15)},
}
btn_ele = ui.TextButton2D(
    label="ELEPHANT",
    size=(90, 26),
    position=(110, 68),
    font_size=14,
    states=btn_states_ele,
)

btn_states_deer = {
    "hover": {"text": "DEER", "color": (0.3, 0.4, 0.3)},
    "pressed": {"text": "DEER", "color": (0.1, 0.2, 0.1)},
    "default": {"text": "DEER", "color": (0.15, 0.25, 0.15)},
}
btn_deer = ui.TextButton2D(
    label="DEER",
    size=(80, 26),
    position=(210, 68),
    font_size=14,
    states=btn_states_deer,
)


def select_closest_animal_by_species(species):
    candidates = [a for a in state["animals"] if a["species"] == species]
    if len(candidates) > 0:
        target = candidates[0]
        state["selected_animal"] = target["id"]
        print(f"Selected Species Snapping focus to animal ID #{target['id']}")


btn_lion.on_clicked = lambda event: select_closest_animal_by_species("lion")
btn_ele.on_clicked = lambda event: select_closest_animal_by_species("elephant")
btn_deer.on_clicked = lambda event: select_closest_animal_by_species("deer")

info_panel.add_element(btn_lion, (20, 68))
info_panel.add_element(btn_ele, (110, 68))
info_panel.add_element(btn_deer, (210, 68))

lbl_card_title = ui.TextBlock2D(
    text="SELECTED ANIMAL STATUS",
    position=(20, 108),
    font_size=13,
    color=(0.95, 0.8, 0.2),
    bold=True,
    dynamic_bbox=True,
)
info_panel.add_element(lbl_card_title, (20, 108))

# Two columns TextBlock2D info
lbl_animal_info_left = ui.TextBlock2D(
    text="Click an animal\nto monitor.",
    position=(20, 132),
    font_size=14,
    color=(0.85, 0.85, 0.85),
    dynamic_bbox=True,
)
info_panel.add_element(lbl_animal_info_left, (20, 132))

lbl_animal_info_right = ui.TextBlock2D(
    text="",
    position=(165, 132),
    font_size=14,
    color=(0.85, 0.85, 0.85),
    dynamic_bbox=True,
)
info_panel.add_element(lbl_animal_info_right, (165, 132))

scene.add(info_panel)


# Controls Panel - Now a TabUI container separating factors
control_panel = ui.TabUI(
    position=(15, 235),
    size=(340, 700),
    tab_titles=["Global", "Lion", "Elephant", "Deer"],
    startup_tab_id=0,
    font_size=16,
    active_color=(0.2, 0.5, 0.8),
    inactive_color=(0.3, 0.3, 0.3),
)

# ----------------- TAB 0: GLOBAL FACTORS -----------------
lbl_g_title = ui.TextBlock2D(
    text="GLOBAL TUNING FACTORS",
    position=(20, 15),
    font_size=15,
    color=(0.95, 0.8, 0.2),
    bold=True,
    dynamic_bbox=True,
)
control_panel.add_element(0, lbl_g_title, (20, 15))

start_y = 70
y_step = 45
current_y = start_y

slider_g_speed = ui.LineSlider2D(
    position=(20, current_y),
    initial_value=1.0,
    min_value=0.0,
    max_value=3.0,
    length=180,
    text_template="Speed Factor: {value:.1f}x",
)
control_panel.add_element(0, slider_g_speed, (20, current_y))
current_y += y_step

slider_g_hunger = ui.LineSlider2D(
    position=(20, current_y),
    initial_value=1.0,
    min_value=0.0,
    max_value=3.0,
    length=180,
    text_template="Hunger Factor: {value:.1f}x",
)
control_panel.add_element(0, slider_g_hunger, (20, current_y))
current_y += y_step

slider_g_thirst = ui.LineSlider2D(
    position=(20, current_y),
    initial_value=1.0,
    min_value=0.0,
    max_value=3.0,
    length=180,
    text_template="Thirst Factor: {value:.1f}x",
)
control_panel.add_element(0, slider_g_thirst, (20, current_y))
current_y += y_step

slider_g_age = ui.LineSlider2D(
    position=(20, current_y),
    initial_value=1.0,
    min_value=0.0,
    max_value=3.0,
    length=180,
    text_template="Age Factor: {value:.1f}x",
)
control_panel.add_element(0, slider_g_age, (20, current_y))
current_y += y_step

slider_g_mating = ui.LineSlider2D(
    position=(20, current_y),
    initial_value=1.0,
    min_value=0.0,
    max_value=3.0,
    length=180,
    text_template="Mating Factor: {value:.1f}x",
)
control_panel.add_element(0, slider_g_mating, (20, current_y))
current_y += y_step

slider_g_hunting = ui.LineSlider2D(
    position=(20, current_y),
    initial_value=1.0,
    min_value=0.0,
    max_value=3.0,
    length=180,
    text_template="Hunting Factor: {value:.1f}x",
)
control_panel.add_element(0, slider_g_hunting, (20, current_y))
current_y += y_step

lbl_g_sim = ui.TextBlock2D(
    text="SIMULATION RATE CONTROLS",
    position=(20, current_y),
    font_size=15,
    color=(0.95, 0.8, 0.2),
    bold=True,
    dynamic_bbox=True,
)
control_panel.add_element(0, lbl_g_sim, (20, current_y))
current_y += 70

slider_global_coh = ui.LineSlider2D(
    position=(20, current_y),
    initial_value=1.0,
    min_value=0.0,
    max_value=3.0,
    length=180,
    text_template="Cohesion Mult: {value:.1f}",
)
control_panel.add_element(0, slider_global_coh, (20, current_y))
current_y += y_step

slider_global_sep = ui.LineSlider2D(
    position=(20, current_y),
    initial_value=1.0,
    min_value=0.0,
    max_value=3.0,
    length=180,
    text_template="Separation Mult: {value:.1f}",
)
control_panel.add_element(0, slider_global_sep, (20, current_y))
current_y += y_step

slider_global_speed = ui.LineSlider2D(
    position=(20, current_y),
    initial_value=1.0,
    min_value=1.0,
    max_value=10.0,
    length=180,
    text_template="Sim Speed: {value:.1f}x",
)
control_panel.add_element(0, slider_global_speed, (20, current_y))
current_y += y_step

# Reset button
btn_states_reset = {
    "hover": {"text": "RESET ALL FACTORS", "color": (0.5, 0.2, 0.2)},
    "pressed": {"text": "RESETTING...", "color": (0.3, 0.1, 0.1)},
    "default": {"text": "RESET ALL FACTORS", "color": (0.4, 0.15, 0.15)},
}
btn_reset = ui.TextButton2D(
    label="RESET ALL FACTORS",
    size=(250, 25),
    position=(20, current_y),
    states=btn_states_reset,
)
control_panel.add_element(0, btn_reset, (20, current_y))


# Global callbacks
def on_g_speed(slider):
    state["global_speed_factor"] = slider.value


def on_g_hunger(slider):
    state["global_hunger_factor"] = slider.value


def on_g_thirst(slider):
    state["global_thirst_factor"] = slider.value


def on_g_age(slider):
    state["global_age_factor"] = slider.value


def on_g_mating(slider):
    state["global_mating_factor"] = slider.value


def on_g_hunting(slider):
    state["global_hunting_factor"] = slider.value


def on_global_coh_change(slider):
    state["cohesion_mult"] = slider.value


def on_global_sep_change(slider):
    state["separation_mult"] = slider.value


def on_global_speed_change(slider):
    state["sim_speed"] = slider.value


slider_g_speed.on_change = on_g_speed
slider_g_hunger.on_change = on_g_hunger
slider_g_thirst.on_change = on_g_thirst
slider_g_age.on_change = on_g_age
slider_g_mating.on_change = on_g_mating
slider_g_hunting.on_change = on_g_hunting
slider_global_coh.on_change = on_global_coh_change
slider_global_sep.on_change = on_global_sep_change
slider_global_speed.on_change = on_global_speed_change

# Keep references to species specific sliders so we can reset their values
species_sliders = []


# Helper to populate species specific factor sliders
def build_species_tab(tab_idx, species_name):
    lbl_title = ui.TextBlock2D(
        text=f"{species_name.upper()} FACTOR ADJUSTERS",
        position=(20, 15),
        font_size=15,
        color=(0.95, 0.8, 0.2),
        bold=True,
        dynamic_bbox=True,
    )
    control_panel.add_element(tab_idx, lbl_title, (20, 15))

    start_y = 70
    y_step = 45
    current_y = start_y

    s_speed = ui.LineSlider2D(
        position=(20, current_y),
        initial_value=1.0,
        min_value=0.0,
        max_value=3.0,
        length=180,
        text_template="Speed Factor: {value:.1f}x",
    )
    control_panel.add_element(tab_idx, s_speed, (20, current_y))
    current_y += y_step

    s_hunger = ui.LineSlider2D(
        position=(20, current_y),
        initial_value=1.0,
        min_value=0.0,
        max_value=3.0,
        length=180,
        text_template="Hunger Factor: {value:.1f}x",
    )
    control_panel.add_element(tab_idx, s_hunger, (20, current_y))
    current_y += y_step

    s_thirst = ui.LineSlider2D(
        position=(20, current_y),
        initial_value=1.0,
        min_value=0.0,
        max_value=3.0,
        length=180,
        text_template="Thirst Factor: {value:.1f}x",
    )
    control_panel.add_element(tab_idx, s_thirst, (20, current_y))
    current_y += y_step

    s_age = ui.LineSlider2D(
        position=(20, current_y),
        initial_value=1.0,
        min_value=0.0,
        max_value=3.0,
        length=180,
        text_template="Age Factor: {value:.1f}x",
    )
    control_panel.add_element(tab_idx, s_age, (20, current_y))
    current_y += y_step

    s_mating = ui.LineSlider2D(
        position=(20, current_y),
        initial_value=1.0,
        min_value=0.0,
        max_value=3.0,
        length=180,
        text_template="Mating Factor: {value:.1f}x",
    )
    control_panel.add_element(tab_idx, s_mating, (20, current_y))
    current_y += y_step

    s_hunting = ui.LineSlider2D(
        position=(20, current_y),
        initial_value=1.0,
        min_value=0.0,
        max_value=3.0,
        length=180,
        text_template="Hunting Factor: {value:.1f}x",
    )
    control_panel.add_element(tab_idx, s_hunting, (20, current_y))

    # Event bindings
    s_speed.on_change = lambda sl: state.update(
        {f"{species_name}_speed_factor": sl.value}
    )
    s_hunger.on_change = lambda sl: state.update(
        {f"{species_name}_hunger_factor": sl.value}
    )
    s_thirst.on_change = lambda sl: state.update(
        {f"{species_name}_thirst_factor": sl.value}
    )
    s_age.on_change = lambda sl: state.update({f"{species_name}_age_factor": sl.value})
    s_mating.on_change = lambda sl: state.update(
        {f"{species_name}_mating_factor": sl.value}
    )
    s_hunting.on_change = lambda sl: state.update(
        {f"{species_name}_hunting_factor": sl.value}
    )

    # Track for reset action
    species_sliders.append(s_speed)
    species_sliders.append(s_hunger)
    species_sliders.append(s_thirst)
    species_sliders.append(s_age)
    species_sliders.append(s_mating)
    species_sliders.append(s_hunting)


# Reset All Factors Callback
def reset_all_factors(event):
    state["global_speed_factor"] = 1.0
    state["global_hunger_factor"] = 1.0
    state["global_thirst_factor"] = 1.0
    state["global_age_factor"] = 1.0
    state["global_mating_factor"] = 1.0
    state["global_hunting_factor"] = 1.0

    state["cohesion_mult"] = 1.0
    state["separation_mult"] = 1.0
    state["sim_speed"] = 1.0

    for name in ["lion", "elephant", "deer"]:
        state[f"{name}_speed_factor"] = 1.0
        state[f"{name}_hunger_factor"] = 1.0
        state[f"{name}_thirst_factor"] = 1.0
        state[f"{name}_age_factor"] = 1.0
        state[f"{name}_mating_factor"] = 1.0
        state[f"{name}_hunting_factor"] = 1.0

    # Sync UI Sliders
    slider_g_speed.value = 1.0
    slider_g_hunger.value = 1.0
    slider_g_thirst.value = 1.0
    slider_g_age.value = 1.0
    slider_g_mating.value = 1.0
    slider_g_hunting.value = 1.0
    slider_global_coh.value = 1.0
    slider_global_sep.value = 1.0
    slider_global_speed.value = 1.0

    for sl in species_sliders:
        sl.value = 1.0


btn_reset.on_clicked = reset_all_factors

# Build Lion, Elephant, Deer Tabs
build_species_tab(1, "lion")
build_species_tab(2, "elephant")
build_species_tab(3, "deer")

scene.add(control_panel)


# Click selection logic
def on_click(event):
    target = event.target
    if hasattr(target, "agent_idx"):
        state["selected_animal"] = target.agent_idx
        print(f"Selected Animal ID #{target.agent_idx}")
    elif target is ground or target is lake:
        state["selected_animal"] = None


# Keyboard input handlers for flying
def on_key_down(event):
    state["keys"].add(event.key.lower())


def on_key_up(event):
    if event.key.lower() in state["keys"]:
        state["keys"].remove(event.key.lower())


# Drag input handlers for free look
def on_pointer_down(event):
    if event.button == 1:
        if (
            not hasattr(event.target, "agent_idx")
            and event.target is not control_panel
            and event.target is not info_panel
        ):
            state["is_dragging_cam"] = True
            state["last_mouse"] = (event.x, event.y)


def on_pointer_move(event):
    if state["is_dragging_cam"] and state["selected_animal"] is None:
        if state["last_mouse"] is not None:
            dx = event.x - state["last_mouse"][0]
            dy = event.y - state["last_mouse"][1]
            state["last_mouse"] = (event.x, event.y)

            # Update orientation
            state["cam_yaw"] -= dx * 0.003
            state["cam_pitch"] = np.clip(
                state["cam_pitch"] - dy * 0.003, -np.pi / 2.2, np.pi / 2.2
            )


def on_pointer_up(event):
    state["is_dragging_cam"] = False
    state["last_mouse"] = None


# Main simulation loop callback
def sim_tick(showm):
    global pos, vel
    dt = 0.16 * state["sim_speed"]

    animals = state["animals"]
    state["screen_size"] = showm.renderer.logical_size

    # Position TabUI dynamically relative to window size on the right side
    tx_pos = state["screen_size"][0] - 335.0
    ty_pos = 15.0
    control_panel.set_position((tx_pos, ty_pos))

    # 1. Animal Needs & Aging Update loop
    for a in animals:
        d_lake = np.linalg.norm(a["pos"])
        in_water = d_lake <= LAKE_RADIUS

        sp = a["species"]
        a["age"] += dt * 0.008 * state["global_age_factor"] * state[f"{sp}_age_factor"]

        h_rate = 0.8 if sp == "lion" else 1.6
        t_rate = 1.2 if sp == "lion" else 2.2

        # Quench thirst if in water, otherwise build up thirst
        if in_water:
            a["thirst"] = max(0.0, a["thirst"] - dt * 15.0)
        else:
            a["thirst"] += (
                dt
                * t_rate
                * state["global_thirst_factor"]
                * state[f"{sp}_thirst_factor"]
            )

        # Build up hunger for all species (reduced later via grazing or hunting)
        a["hunger"] += (
            dt * h_rate * state["global_hunger_factor"] * state[f"{sp}_hunger_factor"]
        )

        # Update mating cooldown
        if a["cooldown"] > 0.0:
            a["cooldown"] -= dt

        # Growth scaling for children
        if a["is_child"]:
            scale = 0.4 + min(0.6, a["age"] * 0.05)
            a["actor"].local.scale = [scale, scale, scale]
            if a["age"] > 12.0:
                a["is_child"] = False

        # Apply starvation/dehydration damage
        if a["hunger"] >= 100.0 or a["thirst"] >= 100.0:
            a["health"] -= dt * 4.0
        else:
            a["health"] = min(MAX_HEALTH[sp], a["health"] + dt * 1.5)

    # Filter dead animals
    dead_list = []
    alive_list = []
    num_lions = 0
    num_elephants = 0
    num_deers = 0

    for a in animals:
        limit_age = MAX_AGE[a["species"]]
        if a["health"] <= 0.0 or a["age"] >= limit_age:
            dead_list.append(a)
        else:
            alive_list.append(a)
            if a["species"] == "lion":
                num_lions += 1
            elif a["species"] == "elephant":
                num_elephants += 1
            else:
                num_deers += 1

    # Remove dead actors
    for a in dead_list:
        scene.remove(a["actor"])
        if state["selected_animal"] == a["id"]:
            state["selected_animal"] = None

    state["animals"] = alive_list
    animals = state["animals"]

    # Limit population explosion
    if len(animals) > 200:
        for a in animals:
            a["cooldown"] = 12.0

    # 2. Ecosystem Behaviors, Steer Forces & Movement
    new_spawns = []

    for a in animals:
        sp = a["species"]
        # Base movement speed scaled by speed factors
        current_max_speed = (
            BASE_SPEED[sp] * state["global_speed_factor"] * state[f"{sp}_speed_factor"]
        )
        run_speed_val = (
            RUN_SPEED[sp] * state["global_speed_factor"] * state[f"{sp}_speed_factor"]
        )

        if a["is_child"]:
            current_max_speed *= 0.6
            run_speed_val *= 0.6

        # Determine priorities
        is_thirsty = a["thirst"] > 45.0
        is_hungry = a["hunger"] > 40.0

        # Species Grouping / Flocking dynamics
        same_species = [
            other
            for other in animals
            if other["species"] == sp and other["id"] != a["id"]
        ]

        flock_cohesion = np.zeros(3)
        flock_alignment = np.zeros(3)
        flock_separation = np.zeros(3)

        if len(same_species) > 0:
            positions = np.array([other["pos"] for other in same_species])
            velocities = np.array([other["vel"] for other in same_species])
            diffs = a["pos"] - positions
            dists_sq = np.sum(diffs**2, axis=-1)

            # Neighbor masks
            neigh_mask = dists_sq < 60.0**2
            sep_mask = dists_sq < 12.0**2

            n_neighbors = np.sum(neigh_mask)
            n_sep = np.sum(sep_mask)

            if n_neighbors > 0:
                avg_pos = np.mean(positions[neigh_mask], axis=0)
                flock_cohesion = avg_pos - a["pos"]
                flock_alignment = np.mean(velocities[neigh_mask], axis=0) - a["vel"]

            if n_sep > 0:
                flock_separation = np.sum(
                    diffs[sep_mask] / (dists_sq[sep_mask][:, None] + 1e-5), axis=0
                )

        flock_weight = 1.0
        if is_thirsty or is_hungry:
            flock_weight = 0.1

        coeffs = GROUP_COEFFS[sp]
        steer = (
            flock_cohesion * coeffs["cohesion"] * 0.1 * state["cohesion_mult"]
            + flock_alignment * coeffs["alignment"] * 0.1
        ) * flock_weight + flock_separation * coeffs["separation"] * 0.3 * state[
            "separation_mult"
        ]

        # Lake Seeking (Thirst)
        if is_thirsty:
            to_lake = LAKE_CENTER - a["pos"]
            d_lake = np.linalg.norm(to_lake)
            if d_lake > LAKE_RADIUS:
                steer += (to_lake / (d_lake + 1e-5)) * 2.8

        # Grazing (Deers and Elephants feeding on vegetation patches)
        if sp in ["deer", "elephant"] and is_hungry:
            best_patch = veg_patches[0]
            min_d = np.linalg.norm(a["pos"] - best_patch)
            for vp in veg_patches[1:]:
                dist = np.linalg.norm(a["pos"] - vp)
                if dist < min_d:
                    min_d = dist
                    best_patch = vp

            to_patch = best_patch - a["pos"]
            to_patch[1] = 0.0
            steer += (to_patch / (min_d + 1e-5)) * 2.2
            if min_d < 25.0:
                a["hunger"] = max(0.0, a["hunger"] - dt * 15.0)

        # Hunting state & pack coordination (Lions targeting Deers)
        if sp == "lion" and is_hungry:
            target_deer = None
            min_d = 9999.0

            pack_target_deer = None
            pack_members = [
                other
                for other in same_species
                if np.linalg.norm(other["pos"] - a["pos"]) < 40.0
                and other["hunt_target_id"] is not None
            ]

            if len(pack_members) > 0:
                shared_id = pack_members[0]["hunt_target_id"]
                for target in animals:
                    if target["id"] == shared_id and target["species"] == "deer":
                        pack_target_deer = target
                        min_d = np.linalg.norm(a["pos"] - target["pos"])
                        break

            if pack_target_deer is not None:
                target_deer = pack_target_deer
            else:
                for target in animals:
                    if target["species"] == "deer":
                        dist = np.linalg.norm(a["pos"] - target["pos"])
                        if dist < min_d:
                            min_d = dist
                            target_deer = target

            if target_deer is not None:
                a["hunt_target_id"] = target_deer["id"]
                current_max_speed = run_speed_val
                to_prey = target_deer["pos"] - a["pos"]
                to_prey[1] = 0.0
                steer += (to_prey / (min_d + 1e-5)) * 3.8

                # Attack/Deal damage over time
                if min_d < 3.5:
                    damage = (
                        dt
                        * 45.0
                        * state["global_hunting_factor"]
                        * state["lion_hunting_factor"]
                    )
                    target_deer["health"] -= damage
                    # Reduce hunger while eating
                    a["hunger"] = max(0.0, a["hunger"] - damage * 1.5)
                    if target_deer["health"] <= 0.0:
                        a["hunger"] = 0.0
                        a["hunt_target_id"] = None
            else:
                a["hunt_target_id"] = None
        else:
            a["hunt_target_id"] = None

        # Predator Avoidance (Deers running from Lions)
        if sp == "deer":
            closest_lion = None
            min_d = 9999.0
            for target in animals:
                if target["species"] == "lion":
                    dist = np.linalg.norm(target["pos"] - a["pos"])
                    if dist < min_d:
                        min_d = dist
                        closest_lion = target

            if closest_lion is not None and min_d < 45.0:
                current_max_speed = run_speed_val
                to_pred = closest_lion["pos"] - a["pos"]
                to_pred[1] = 0.0
                steer -= (to_pred / (min_d + 1e-5)) * 3.8

        # Lions avoid Elephants
        if sp == "lion":
            for target in animals:
                if target["species"] == "elephant":
                    dist = np.linalg.norm(target["pos"] - a["pos"])
                    if dist < 28.0:
                        current_max_speed = run_speed_val
                        steer -= (target["pos"] - a["pos"]) / (dist + 1e-5) * 3.8

        # Elephant Family Defense & Retaliation against Lions
        if sp == "elephant":
            is_rage = False
            target_lion = None
            if not a["is_child"]:
                for lion in animals:
                    if lion["species"] == "lion":
                        dist_lion = np.linalg.norm(lion["pos"] - a["pos"])
                        if dist_lion < 35.0:
                            is_rage = True
                            target_lion = lion
                            break

            if is_rage and target_lion is not None:
                current_max_speed = run_speed_val
                to_lion = target_lion["pos"] - a["pos"]
                to_lion[1] = 0.0
                dist = np.linalg.norm(to_lion)
                steer += (to_lion / (dist + 1e-5)) * 4.2
                if dist < 4.5:
                    target_lion["health"] -= dt * 90.0

        is_satiated = (not is_hungry) and (not is_thirsty)
        if is_satiated:
            current_max_speed *= 0.3  # chill out / wander slowly

        # Breeding Logic
        if is_satiated and a["cooldown"] <= 0.0 and not a["is_child"]:
            for partner in animals:
                if (
                    partner["species"] == sp
                    and partner["id"] != a["id"]
                    and partner["gender"] != a["gender"]
                    and partner["cooldown"] <= 0.0
                    and not partner["is_child"]
                ):
                    dist = np.linalg.norm(a["pos"] - partner["pos"])
                    if dist < 12.0:
                        cooldown_val = 35.0 / (
                            state["global_mating_factor"] * state[f"{sp}_mating_factor"]
                            + 1e-5
                        )
                        a["cooldown"] = cooldown_val
                        partner["cooldown"] = cooldown_val
                        mid_pos = (a["pos"] + partner["pos"]) * 0.5
                        mid_pos[1] = 0.0
                        new_spawns.append((sp, mid_pos))
                        break

        # Map Boundaries Constraint
        dist_from_origin = np.linalg.norm(a["pos"])
        if dist_from_origin > 280.0:
            steer -= (a["pos"] / (dist_from_origin + 1e-5)) * 3.5

        # Update velocity & limit speed
        a["vel"] += steer * dt * 22.0
        speed = np.linalg.norm(a["vel"])
        if speed > current_max_speed:
            a["vel"] = (a["vel"] / speed) * current_max_speed
        elif speed < MIN_SPEED:
            a["vel"] = (a["vel"] / (speed + 1e-5)) * MIN_SPEED

        # Position update
        a["pos"] += a["vel"] * dt

        # Hard Boundary constraint: keep inside boundary radius of 290
        d_origin = np.linalg.norm(a["pos"])
        if d_origin > 290.0:
            a["pos"] = (a["pos"] / (d_origin + 1e-5)) * 290.0
            a["vel"] = -(a["pos"] / 290.0) * np.linalg.norm(a["vel"])

        # Project animal pos dynamically to follow terrain elevation changes
        h_offset = 0.6 if sp == "lion" else 0.5
        if sp == "elephant":
            h_offset = 1.0
        a["pos"][1] = get_terrain_height(a["pos"][0], a["pos"][2]) + h_offset

        # Update actor transforms
        a["actor"].local.position = a["pos"]
        h = a["vel"] / (np.linalg.norm(a["vel"]) + 1e-5)
        angle = np.arctan2(h[0], h[2])
        s = np.sin(angle / 2.0)
        c = np.cos(angle / 2.0)
        a["actor"].local.rotation = [0.0, s, 0.0, c]

    # Spawn new babies
    for species, position in new_spawns:
        spawn_animal(species, position, is_child=True)

    # 3. UI Status updates
    lbl_legend.message = (
        f"Lions: {num_lions} | Elephants: {num_elephants} | Deers: {num_deers}"
    )

    selected = state["selected_animal"]
    camera = showm.screens[0].camera

    # Update highlighting
    if selected is not None:
        for a in animals:
            if a["id"] == selected:
                a["actor"].color = ANIMAL_COLOR[a["species"]]
            else:
                a["actor"].color = tuple(np.array(ANIMAL_COLOR[a["species"]]) * 0.3)
    else:
        for a in animals:
            a["actor"].color = ANIMAL_COLOR[a["species"]]

    # 4. Camera Control (Focus vs Free look Fly mode)
    if selected is not None:
        selected_a = None
        for a in animals:
            if a["id"] == selected:
                selected_a = a
                break

        if selected_a is not None:
            gender_txt = "Male" if selected_a["gender"] == "M" else "Female"
            type_txt = "Calf" if selected_a["is_child"] else "Adult"
            if selected_a["species"] == "deer":
                type_txt = "Fawn" if selected_a["is_child"] else "Adult"
            elif selected_a["species"] == "lion":
                type_txt = "Cub" if selected_a["is_child"] else "Adult"

            # Display stats on details card
            # Display stats on details card (Two columns)
            lbl_animal_info_left.message = (
                f"Species: {selected_a['species'].upper()}\n"
                f"Class: {type_txt}\n"
                f"Sex: {gender_txt}\n"
                f"Health: {selected_a['health']:.1f}"
            )
            lbl_animal_info_right.message = (
                f"Hunger: {selected_a['hunger']:.1f}%\n"
                f"Thirst: {selected_a['thirst']:.1f}%\n"
                f"Age: {selected_a['age']:.1f}\n"
                f"CD: {selected_a['cooldown']:.1f}s"
            )

            # Chase camera positioning
            b_pos = selected_a["pos"]
            b_vel = selected_a["vel"]
            spd = np.linalg.norm(b_vel)
            h = b_vel / (spd + 1e-5)

            target_cam_pos = b_pos - h * 25.0 + np.array([0.0, 12.0, 0.0])
            camera.local.position = (
                camera.local.position + (target_cam_pos - camera.local.position) * 0.1
            )
            camera.look_at(b_pos + h * 4.0)

            # Update camera look rotation states to match
            diff = b_pos - camera.local.position
            state["cam_yaw"] = np.arctan2(diff[0], diff[2])
            state["cam_pitch"] = np.arcsin(
                np.clip(diff[1] / (np.linalg.norm(diff) + 1e-5), -0.99, 0.99)
            )
        else:
            state["selected_animal"] = None
            lbl_animal_info_left.message = "Click an animal\nto monitor."
            lbl_animal_info_right.message = ""
    else:
        lbl_animal_info_left.message = "Click an animal\nto monitor."
        lbl_animal_info_right.message = ""

        keys = state["keys"]

        # Keyboard Camera Rotation
        if "arrowleft" in keys or "left" in keys:
            state["cam_yaw"] += 1.5 * dt
        if "arrowright" in keys or "right" in keys:
            state["cam_yaw"] -= 1.5 * dt
        if "arrowup" in keys or "up" in keys:
            state["cam_pitch"] = np.clip(
                state["cam_pitch"] + 1.5 * dt, -np.pi / 2.2, np.pi / 2.2
            )
        if "arrowdown" in keys or "down" in keys:
            state["cam_pitch"] = np.clip(
                state["cam_pitch"] - 1.5 * dt, -np.pi / 2.2, np.pi / 2.2
            )

        # Counterstrike style WASD fly camera mode
        qy = axis_angle_to_quat(np.array([0, 1, 0]), np.degrees(state["cam_yaw"]))
        qx = axis_angle_to_quat(np.array([1, 0, 0]), np.degrees(state["cam_pitch"]))
        camera.local.rotation = quat_mult(qy, qx)

        fwd = rotate_vector(camera.local.rotation, np.array([0.0, 0.0, -1.0]))
        right = rotate_vector(camera.local.rotation, np.array([1.0, 0.0, 0.0]))

        fly_speed = 70.0
        move = np.zeros(3)

        if "w" in keys:
            move += fwd
        if "s" in keys:
            move -= fwd
        if "a" in keys:
            move -= right
        if "d" in keys:
            move += right
        if "q" in keys:
            move += np.array([0.0, 1.0, 0.0])
        if "e" in keys:
            move -= np.array([0.0, 1.0, 0.0])

        if np.any(move):
            move_dir = move / np.linalg.norm(move)
            camera.local.position = camera.local.position + move_dir * fly_speed * dt

    # Enforce camera does not move below ground
    cx, cy, cz = camera.local.position
    cam_terrain_h = get_terrain_height(cx, cz)
    cy_new = max(cy, cam_terrain_h + 5.0)
    camera.local.position = np.array([cx, cy_new, cz])

    showm.render()


if __name__ == "__main__":
    show_manager = window.ShowManager(
        scene=scene, size=(1024, 768), title="Jungle Ecosystem Survival Simulator"
    )

    # Disable default camera controller to allow our WASD mouse free-look camera
    show_manager.screens[0].controller.enabled = False

    # Bind pointer clicks and keyboard fly controls
    show_manager.renderer.add_event_handler(on_click, EventType.POINTER_DOWN)
    show_manager.renderer.add_event_handler(on_key_down, EventType.KEY_DOWN)
    show_manager.renderer.add_event_handler(on_key_up, EventType.KEY_UP)
    show_manager.renderer.add_event_handler(on_pointer_down, EventType.POINTER_DOWN)
    show_manager.renderer.add_event_handler(on_pointer_move, EventType.POINTER_MOVE)
    show_manager.renderer.add_event_handler(on_pointer_up, EventType.POINTER_UP)

    # Initial camera placement
    camera = show_manager.screens[0].camera
    camera.local.position = (0.0, 200.0, -250.0)

    # Set initial camera look rotation
    state["cam_yaw"] = 0.0
    state["cam_pitch"] = -np.radians(40.0)
    qy = axis_angle_to_quat(np.array([0, 1, 0]), np.degrees(state["cam_yaw"]))
    qx = axis_angle_to_quat(np.array([1, 0, 0]), np.degrees(state["cam_pitch"]))
    camera.local.rotation = quat_mult(qy, qx)

    # Ensure UI elements always render on top
    disable_depth_testing(scene.ui_scene)

    # Start simulation
    show_manager.register_callback(sim_tick, 0.016, True, "JungleLoop", show_manager)
    show_manager.start()
