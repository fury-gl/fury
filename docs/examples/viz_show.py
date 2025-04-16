import numpy as np

from fury.window import show
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
    show(actors=[sphere_actor])
