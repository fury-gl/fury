import argparse
import numpy as np

from fury.window import ShowManager, Scene, snapshot
from fury.actor import sphere
from fury.data import read_viz_cubemap, fetch_viz_cubemaps
from fury.io import load_cube_map_texture


###############################################################################
# Let's fetch a skybox texture from the FURY data repository.

fetch_viz_cubemaps()

###############################################################################
# The following function returns the full path of the 6 images composing the
# skybox.

texture_files = read_viz_cubemap("skybox")

###############################################################################
# Now that we have the location of the textures, let's load them and create a
# Cube Map Texture object.

cube_map = load_cube_map_texture(texture_files)

###############################################################################
# The Scene object in FURY can handle cube map textures and extract light
# information from them, so it can be used to create more plausible materials
# interactions. The ``skybox`` parameter takes as input a cube map texture and
# performs the previously described process.

scene0 = Scene(skybox=cube_map)
scene1 = Scene(background=(1, 1, 1, 1))
scene2 = Scene(background=(1, 0, 0, 1))

###############################################################################
# Let's create three different sphere actors to add to respective scenes.
# Note: Adding same actor to multiple scenes will not work and only add to the
# last scene that got the actor added to it.

sphere_actor0 = sphere(
    np.zeros((1, 3)),
    colors=(1, 0, 1, 1),
    radii=15.0,
    phi=48,
    theta=48,
)
sphere_actor1 = sphere(
    np.zeros((1, 3)),
    colors=(1, 0, 1, 1),
    radii=15.0,
    phi=48,
    theta=48,
)
sphere_actor2 = sphere(
    np.zeros((1, 3)),
    colors=(1, 0, 1, 1),
    radii=15.0,
    phi=48,
    theta=48,
)

scene0.add(sphere_actor0)
scene1.add(sphere_actor1)
scene2.add(sphere_actor2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Enable interactive mode"
    )
    args = parser.parse_args()

    if args.interactive:
        show_m = ShowManager(
            scene=[scene0, scene1, scene2],
            title="FURY 2.0: Multi Screen Example",
            screen_config=[2, 1],
        )
        show_m.start()
    else:
        snapshot(
            scene=[scene0, scene1, scene2],
            fname="multi_screen.png",
            screen_config=[2, 1],
        )
