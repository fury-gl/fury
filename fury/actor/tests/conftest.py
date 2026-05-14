import numpy as np
import pytest
import rendercanvas.glfw

from fury import actor
from fury.primitive import prim_sphere


def _do_nothing_patch(self):
    pass


rendercanvas.glfw.RenderCanvas._rc_close = _do_nothing_patch


@pytest.fixture
def sphere_actor():
    centers = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    return actor.sphere(centers=centers, colors=colors)


@pytest.fixture
def sphere_prim():
    vertices, faces = prim_sphere(phi=8, theta=8)
    return vertices, faces
