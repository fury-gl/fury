import numpy as np
import argparse

from fury.window import show, snapshot
from fury.actor import sphere

###############################################################################
# Let's create sphere actor to add three spheres to display.

sphere_actor = sphere(
    np.asarray([(15, 0, 0), (0, 15, 0), (0, 0, 15)]).reshape((3, 3)),
    radii=15,
    colors=np.asarray([(1, 0, 0), (0, 1, 0), (0, 0, 1)]).reshape((3, 3)),
    phi=48,
    theta=48,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Enable interactive mode"
    )
    args = parser.parse_args()

    if args.interactive:
        show(actors=[sphere_actor])
    else:
        snapshot(actors=[sphere_actor], fname="show.png")
