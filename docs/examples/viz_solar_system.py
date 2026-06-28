"""
=======================
Solar System Animation
=======================

In this tutorial, we will create an animation of the solar system
using textured spheres. We will also show how to manipulate the
position of these sphere actors in a timer_callback function
to simulate orbital motion.
"""

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from fury import actor, window
from fury.primitive import prim_sphere
from fury.ui import PlaybackPanel
from fury.data import fetch_viz_textures, read_viz_textures

##############################################################################
# Create a scene to start.

scene = window.Scene()

##############################################################################
# Define information relevant for each planet actor including its
# texture name, relative position, and scale.

planets_data = [
    {
        "filename": "8k_mercury.jpg",
        "position": 8,
        "earth_days": 58.0,
        "scale": (0.3, 0.3, 0.3),
    },
    {
        "filename": "8k_venus_surface.jpg",
        "position": 10,
        "earth_days": 243.0,
        "scale": (0.76, 0.76, 0.76),
    },
    {
        "filename": "1_earth_8k.jpg",
        "position": 12,
        "earth_days": 1.0,
        "scale": (0.8, 0.8, 0.8),
    },
    {
        "filename": "8k_mars.jpg",
        "position": 14,
        "earth_days": 1.03,
        "scale": (0.42, 0.42, 0.42),
    },
    {
        "filename": "jupiter.jpg",
        "position": 20,
        "earth_days": 0.41,
        "scale": (2.5, 2.5, 2.5),
    },
    {
        "filename": "8k_saturn.jpg",
        "position": 28,
        "earth_days": 0.45,
        "scale": (2.1, 2.1, 2.1),
    },
    {
        "filename": "8k_saturn_ring_alpha.png",
        "position": 28,
        "earth_days": 0.45,
        "scale": (3.15, 0.5, 3.15),
    },
    {
        "filename": "2k_uranus.jpg",
        "position": 38,
        "earth_days": 0.72,
        "scale": (1.3, 1.3, 1.3),
    },
    {
        "filename": "2k_neptune.jpg",
        "position": 49,
        "earth_days": 0.67,
        "scale": (1.25, 1.25, 1.25),
    },
    {
        "filename": "8k_sun.jpg",
        "position": 0,
        "earth_days": 27.0,
        "scale": (6.0, 6.0, 6.0),
    },
]

fetch_viz_textures()

##############################################################################
# Create Textured Spheres and adding to the Scene


def make_textured_sphere(planet_file, scale, position=None):
    verts, faces = prim_sphere(phi=60, theta=60)

    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = verts / norms

    x = normalized[:, 0]
    y = normalized[:, 1]
    z = normalized[:, 2]

    # Standard Mercator mapping:
    # longitude u: [-pi, pi] -> [0, 1]
    # latitude v: [-pi/2, pi/2] -> [0, 1]
    u = np.arctan2(x, z) / (2.0 * np.pi) + 0.5
    v = 0.5 - np.arcsin(y) / np.pi
    uvs = np.column_stack((u, v))

    planet_actor = actor.surface(verts, faces, texture=planet_file, texture_coords=uvs)

    planet_actor.local.scale = scale
    if position is not None:
        planet_actor.local.position = position

    return planet_actor


##############################################################################
# Initialize all planet actors.

planets = []
for p_data in planets_data:
    filename = p_data["filename"]
    pos = p_data["position"]
    earth_days = p_data["earth_days"]
    scale = p_data["scale"]

    planet_file = read_viz_textures(filename)

    initial_pos = [float(pos), 0.0, 0.0]
    actor_obj = make_textured_sphere(planet_file, scale, position=initial_pos)
    scene.add(actor_obj)

    is_ring = "saturn_ring" in filename
    is_sun = "sun" in filename

    p_info = {
        "filename": filename,
        "actor": actor_obj,
        "radius": float(pos),
        "earth_days": float(earth_days),
        "is_ring": is_ring,
        "is_sun": is_sun,
    }
    planets.append(p_info)


##############################################################################
# Define Constants and util functions to calculate orbital paths and positions
# for the visual track lines and planet movements.

g_exponent = np.float_power(10, -11)
g_constant = 6.673 * g_exponent

m_exponent = 1073741824
m_constant = 1.989 * m_exponent

miu = m_constant * g_constant


def get_orbit_period(radius):
    if radius == 0.0:
        return 1.0
    return 2.0 * np.pi * np.sqrt(np.power(radius, 3) / miu)


def get_orbital_position(radius, time):
    if radius == 0.0:
        return 0.0, 0.0
    orbit_period = get_orbit_period(radius)
    x = radius * np.cos((-2.0 * np.pi * time) / orbit_period)
    z = radius * np.sin((-2.0 * np.pi * time) / orbit_period)
    return x, z


r_planets = [
    float(p["radius"]) for p in planets if not p["is_sun"] and not p["is_ring"]
]


def calculate_path(r_planet):
    planet_track = []
    orbit_period = get_orbit_period(r_planet)
    for t in np.linspace(0.0, orbit_period, 200):
        x, z = get_orbital_position(r_planet, t)
        planet_track.append([x, 0.0, z])
    return planet_track


planet_tracks = [calculate_path(rplanet) for rplanet in r_planets]


orbit_actor = actor.line(planet_tracks, colors=(1.0, 1.0, 1.0))
orbit_actor.local.position = (0.0, 0.0, 0.0)
scene.add(orbit_actor)

##############################################################################
# Initialize PlaybackPanel UI.

playback_ui = PlaybackPanel(position=(50, 50), width=700, loop=True)
playback_ui.final_time = 2000.0
scene.add(playback_ui)

state = {"current_time": 0.0, "rotation_speed": 1.0}


def update_planet_transforms(p_info, t):
    actor_obj = p_info["actor"]

    if p_info["is_sun"]:
        angle_axial = (50.0 / p_info["earth_days"]) * t
        R_axial = Rot.from_euler("y", angle_axial, degrees=True)
        actor_obj.local.rotation = R_axial.as_quat()
        actor_obj.local.position = [0.0, 0.0, 0.0]
    else:
        x, z = get_orbital_position(p_info["radius"], t)
        actor_obj.local.position = [x, 0.0, z]

        if p_info["is_ring"]:
            pass
        else:
            angle_axial = (50.0 / p_info["earth_days"]) * t
            R_axial = Rot.from_euler("y", angle_axial, degrees=True)
            actor_obj.local.rotation = R_axial.as_quat()


def on_progress_changed(t):
    state["current_time"] = t
    for p in planets:
        update_planet_transforms(p, t)


def on_speed_changed(s):
    state["rotation_speed"] = s


playback_ui.on_progress_bar_changed = on_progress_changed
playback_ui.on_speed_changed = on_speed_changed


def update_playback_logic(show_manager_obj=None):
    """Callback to sync animation state with PlaybackPanel UI."""
    if playback_ui._playing:
        step = 1.0 * state["rotation_speed"]
        state["current_time"] += step

        if state["current_time"] > playback_ui.final_time:
            if playback_ui._loop:
                state["current_time"] = 0.0
            else:
                state["current_time"] = playback_ui.final_time
                playback_ui.pause()

        playback_ui.current_time = state["current_time"]

        for p in planets:
            update_planet_transforms(p, state["current_time"])


if __name__ == "__main__":
    ##############################################################################
    # Start the ShowManager and Register the Callback.

    showm = window.ShowManager(
        scene=scene, size=(900, 768), title="FURY Solar System Animation"
    )

    camera = showm.screens[0].camera
    camera.local.position = (-30.0, 90.0, 150.0)
    camera.look_at((0.0, 0.0, 0.0))

    showm.register_callback(update_playback_logic, 0.01, True, "PlaybackSync", showm)

    showm.start()

showm.snapshot(fname="viz_solar_system_animation.png")
