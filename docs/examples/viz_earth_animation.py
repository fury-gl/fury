"""Texture Sphere Animation (Earth, Moon, Satellite).

Port of the classic Earth animation tutorial into the v2 branch.

This example:
- Creates textured Earth and Moon spheres
- Adds a marker + text on Earth
- Adds a satellite model around the Moon
- Animates camera + rotations using a timer callback
"""

import itertools

from fury import window, actor, utils, io
from fury.data import (
    read_viz_textures,
    fetch_viz_textures,
    read_viz_models,
    fetch_viz_models,
)


def main(interactive: bool = True) -> None:
    # ---- 1. Scene ----
    scene = window.Scene()

    # ---- 2. Load Earth + Moon textures ----
    fetch_viz_textures()

    earth_filename = read_viz_textures("1_earth_8k.jpg")
    earth_image = io.load_image(earth_filename)
    earth_actor = actor.texture_on_sphere(earth_image)

    moon_filename = read_viz_textures("moon-8k.jpg")
    moon_image = io.load_image(moon_filename)
    moon_actor = actor.texture_on_sphere(moon_image)

    scene.add(earth_actor)
    scene.add(moon_actor)

    # Position + scale the Moon, rotate Earth texture
    moon_actor.SetPosition(1, 0.1, 0.5)
    moon_actor.SetScale(0.25, 0.25, 0.25)
    utils.rotate(earth_actor, (-90, 1, 0, 0))

    # ---- 3. Marker + text on Earth ----
    center = [[-0.39, 0.3175, 0.025]]
    radius = 0.002
    sphere_actor = actor.sphere(center, window.colors.blue_medium, radius)

    text_actor = actor.text_3d(
        "Bloomington, Indiana",
        (-0.42, 0.31, 0.03),
        window.colors.white,
        0.004,
    )
    utils.rotate(text_actor, (-90, 0, 1, 0))

    # ---- 4. Satellite around the Moon ----
    fetch_viz_models()
    satellite_filename = read_viz_models("satellite_obj.obj")
    satellite = io.load_polydata(satellite_filename)
    satellite_actor = utils.get_actor_from_polydata(satellite)

    satellite_actor.SetPosition(-0.75, 0.1, 0.4)
    satellite_actor.SetScale(0.005, 0.005, 0.005)

    # ---- 5. ShowManager + camera ----
    showm = window.ShowManager(
        scene,
        size=(900, 768),
        reset_camera=False,
        order_transparent=True,
    )

    counter = itertools.count()

    scene.set_camera(
        position=(0.24, 0.00, 4.34),
        focal_point=(0.00, 0.00, 0.00),
        view_up=(0.00, 1.00, 0.00),
    )

    # ---- 6. Timer callback for animation ----
    def timer_callback(_obj, _event):
        cnt = next(counter)
        showm.render()

        if cnt < 450:
            utils.rotate(earth_actor, (1, 0, 1, 0))

        if cnt % 5 == 0 and cnt < 450:
            showm.scene.azimuth(-1)

        if cnt == 300:
            scene.set_camera(
                position=(-3.679, 0.00, 2.314),
                focal_point=(0.0, 0.35, 0.00),
                view_up=(0.00, 1.00, 0.00),
            )

        if 300 < cnt < 450:
            scene.zoom(1.01)

        if 450 <= cnt < 1500:
            scene.add(sphere_actor)
            scene.add(text_actor)

        if 450 <= cnt < 550:
            scene.zoom(1.01)

        if cnt == 575:
            moon_actor.SetPosition(-1, 0.1, 0.5)
            scene.set_camera(
                position=(-0.5, 0.1, 0.00),
                focal_point=(-1, 0.1, 0.5),
                view_up=(0.00, 1.00, 0.00),
            )
            scene.zoom(0.03)
            scene.add(satellite_actor)
            utils.rotate(satellite_actor, (180, 0, 1, 0))
            scene.rm(earth_actor)

        if 575 < cnt < 750:
            showm.scene.azimuth(-2)
            utils.rotate(moon_actor, (-2, 0, 1, 0))
            satellite_actor.SetPosition(-0.8, 0.1 - cnt / 10000.0, 0.4)

        if 750 <= cnt < 1100:
            showm.scene.azimuth(-2)
            utils.rotate(moon_actor, (-2, 0, 1, 0))
            satellite_actor.SetPosition(-0.8, -0.07 + cnt / 10000.0, 0.4)

        if cnt == 1100:
            showm.exit()

    showm.initialize()
    showm.add_timer_callback(True, 35, timer_callback)

    if interactive:
        showm.start()

    # Save a frame for the docs gallery
    window.record(
        showm.scene,
        size=(900, 768),
        out_path="viz_earth_animation.png",
    )


if __name__ == "__main__":
    main(interactive=True)
