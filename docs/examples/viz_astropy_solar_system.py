"""
===============================
Astropy Solar System Animation
===============================

In this tutorial, we will create an animation of the solar system
using FURY textured spheres and the astropy library to fetch
accurate astronomical positions of the planets based on a starting date.

Distances have been visually compressed using a scaling factor to keep
the visualization compact while preserving the accurate celestial
inclinations and eccentricities of the real orbits.
"""

import warnings
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from astropy.time import Time
from astropy.coordinates import get_body_barycentric, solar_system_ephemeris

from fury import actor, window
from fury.primitive import prim_sphere
from fury.ui import PlaybackPanel
from fury.data import fetch_viz_textures, read_viz_textures

# Ignore astropy warnings about dubious years when querying far into the future
warnings.filterwarnings("ignore")

##############################################################################
# Create a scene to start.

scene = window.Scene()

##############################################################################
# Define information relevant for each planet actor.
# We include visual radii to compress the visualization (so Neptune isn't
# unseeably far away), and we map each to its astropy body name and
# semi-major axis to calculate a scaling factor that preserves eccentricity
# and orbital inclination accurately.

planets_data = [
    {
        "filename": "8k_mercury.jpg",
        "astropy_name": "mercury",
        "visual_radius": 8.0,
        "semi_major_axis": 0.387,
        "earth_days": 58.0,
        "scale": (0.3, 0.3, 0.3),
        "orbital_period_days": 88.0,
    },
    {
        "filename": "8k_venus_surface.jpg",
        "astropy_name": "venus",
        "visual_radius": 10.0,
        "semi_major_axis": 0.723,
        "earth_days": 243.0,
        "scale": (0.76, 0.76, 0.76),
        "orbital_period_days": 224.7,
    },
    {
        "filename": "1_earth_8k.jpg",
        "astropy_name": "earth",
        "visual_radius": 12.0,
        "semi_major_axis": 1.000,
        "earth_days": 1.0,
        "scale": (0.8, 0.8, 0.8),
        "orbital_period_days": 365.25,
    },
    {
        "filename": "8k_mars.jpg",
        "astropy_name": "mars",
        "visual_radius": 14.0,
        "semi_major_axis": 1.524,
        "earth_days": 1.03,
        "scale": (0.42, 0.42, 0.42),
        "orbital_period_days": 687.0,
    },
    {
        "filename": "jupiter.jpg",
        "astropy_name": "jupiter",
        "visual_radius": 20.0,
        "semi_major_axis": 5.204,
        "earth_days": 0.41,
        "scale": (2.5, 2.5, 2.5),
        "orbital_period_days": 4333.0,
    },
    {
        "filename": "8k_saturn.jpg",
        "astropy_name": "saturn",
        "visual_radius": 28.0,
        "semi_major_axis": 9.582,
        "earth_days": 0.45,
        "scale": (2.1, 2.1, 2.1),
        "orbital_period_days": 10759.0,
    },
    {
        "filename": "8k_saturn_ring_alpha.png",
        "astropy_name": "saturn",
        "visual_radius": 28.0,
        "semi_major_axis": 9.582,
        "earth_days": 0.45,
        "scale": (3.15, 0.5, 3.15),
        "orbital_period_days": 10759.0,
        "is_ring": True,
    },
    {
        "filename": "2k_uranus.jpg",
        "astropy_name": "uranus",
        "visual_radius": 38.0,
        "semi_major_axis": 19.201,
        "earth_days": 0.72,
        "scale": (1.3, 1.3, 1.3),
        "orbital_period_days": 30688.0,
    },
    {
        "filename": "2k_neptune.jpg",
        "astropy_name": "neptune",
        "visual_radius": 49.0,
        "semi_major_axis": 30.047,
        "earth_days": 0.67,
        "scale": (1.25, 1.25, 1.25),
        "orbital_period_days": 60182.0,
    },
    {
        "filename": "8k_sun.jpg",
        "astropy_name": "sun",
        "visual_radius": 0.0,
        "semi_major_axis": 1.0,
        "earth_days": 27.0,
        "scale": (6.0, 6.0, 6.0),
        "orbital_period_days": 1.0,
        "is_sun": True,
    },
]

fetch_viz_textures()


def make_textured_sphere(planet_file, scale, position=None):
    verts, faces = prim_sphere(phi=60, theta=60)

    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = verts / norms

    x = normalized[:, 0]
    y = normalized[:, 1]
    z = normalized[:, 2]

    # Standard Mercator mapping:
    u = np.arctan2(x, z) / (2.0 * np.pi) + 0.5
    v = 0.5 - np.arcsin(y) / np.pi
    uvs = np.column_stack((u, v))

    planet_actor = actor.surface(verts, faces, texture=planet_file, texture_coords=uvs)

    planet_actor.local.scale = scale
    if position is not None:
        planet_actor.local.position = position

    return planet_actor


def get_astropy_position(astropy_name, t_obj, visual_radius, semi_major_axis):
    """Fetch the barycentric position and apply visual compression scaling."""
    if astropy_name == "sun":
        if t_obj.isscalar:
            return 0.0, 0.0, 0.0
        else:
            return np.zeros(len(t_obj)), np.zeros(len(t_obj)), np.zeros(len(t_obj))

    with solar_system_ephemeris.set("builtin"):
        pos = get_body_barycentric(astropy_name, t_obj)
        scale = visual_radius / semi_major_axis
        # Swap Y and Z axes so the orbits lie horizontally in the X-Z plane,
        # matching FURY's default camera view
        return pos.x.value * scale, pos.z.value * scale, pos.y.value * scale


# State variables
start_jd = Time.now().jd
state = {"current_time": 0.0, "rotation_speed": 1.0, "start_jd": start_jd}
# Number of Earth days passed per 1 unit of animation time (UI step)
UI_STEP_TO_DAYS = 2.0

planets = []
for p_data in planets_data:
    filename = p_data["filename"]
    planet_file = read_viz_textures(filename)

    # Calculate initial position based on start date
    x, y, z = get_astropy_position(
        p_data["astropy_name"],
        Time(start_jd, format="jd"),
        p_data["visual_radius"],
        p_data["semi_major_axis"],
    )
    initial_pos = [x, y, z]

    actor_obj = make_textured_sphere(planet_file, p_data["scale"], position=initial_pos)
    scene.add(actor_obj)

    p_info = p_data.copy()
    p_info["actor"] = actor_obj
    p_info["is_ring"] = p_data.get("is_ring", False)
    p_info["is_sun"] = p_data.get("is_sun", False)
    planets.append(p_info)


##############################################################################
# Calculate Orbital Paths
# We draw the orbit for the duration of 1 full orbital period.


def calculate_path(p_info):
    planet_track = []
    period = p_info["orbital_period_days"]
    # Ensure smooth circles for the paths (200 points)
    times = Time(np.linspace(start_jd, start_jd + period, 200), format="jd")

    x_arr, y_arr, z_arr = get_astropy_position(
        p_info["astropy_name"],
        times,
        p_info["visual_radius"],
        p_info["semi_major_axis"],
    )

    for x, y, z in zip(x_arr, y_arr, z_arr, strict=False):
        planet_track.append([x, y, z])

    return planet_track


# We only calculate paths for planets, not the sun or the rings
track_planets = [p for p in planets if not p["is_sun"] and not p["is_ring"]]
planet_tracks = [calculate_path(p) for p in track_planets]

orbit_actor = actor.line(planet_tracks, colors=(1.0, 1.0, 1.0))
orbit_actor.local.position = (0.0, 0.0, 0.0)
scene.add(orbit_actor)

##############################################################################
# Initialize PlaybackPanel UI.

playback_ui = PlaybackPanel(position=(50, 50), width=700, loop=True)
playback_ui.final_time = 2000.0
scene.add(playback_ui)


def update_planet_transforms(p_info, t):
    actor_obj = p_info["actor"]

    if p_info["is_sun"]:
        angle_axial = (50.0 / p_info["earth_days"]) * t
        R_axial = Rot.from_euler("y", angle_axial, degrees=True)
        actor_obj.local.rotation = R_axial.as_quat()
        actor_obj.local.position = [0.0, 0.0, 0.0]
    else:
        current_jd = state["start_jd"] + (t * UI_STEP_TO_DAYS)
        current_time = Time(current_jd, format="jd")
        x, y, z = get_astropy_position(
            p_info["astropy_name"],
            current_time,
            p_info["visual_radius"],
            p_info["semi_major_axis"],
        )
        actor_obj.local.position = [x, y, z]

        if not p_info["is_ring"]:
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
        scene=scene, size=(900, 768), title="FURY Astropy Solar System"
    )

    camera = showm.screens[0].camera
    camera.local.position = (-30.0, 90.0, 150.0)
    camera.look_at((0.0, 0.0, 0.0))

    showm.register_callback(update_playback_logic, 0.01, True, "PlaybackSync", showm)

    showm.start()
    showm.snapshot(fname="viz_astropy_solar_system.png")
