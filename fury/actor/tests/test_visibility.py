"""
Visibility snapshot tests for every renderable actor.

Each actor is built with real inputs (numpy arrays / values -- no mocks,
patches, or dummy classes), rendered offscreen, and analyzed with
:func:`fury.window.analyze_snapshot`. The helper asserts the actor produces
foreground pixels when visible and nothing when hidden, toggling through the
actor's real ``.visible`` attribute (which is exactly what
:func:`fury.actor.set_group_visibility` does for a group).
"""

import numpy as np
import pytest

from fury import actor
from fury.actor.tests._helpers import assert_visibility

# --- Shared real inputs -----------------------------------------------------

CENTERS = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
LINES = [np.array([[0, 0, 0], [1, 1, 1], [2, 0, 0]], dtype=np.float32)]


def _gradient_image(n=64):
    """A non-uniform 2D image so the rendered frame is never flat."""
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.outer(ramp, ramp)


def _scalar_volume(n=32):
    """A non-uniform 3D scalar volume (deterministic)."""
    x, y, z = np.mgrid[0:n, 0:n, 0:n]
    return ((x + y + z) % 17).astype(np.float32)


def _roi_volume(n=20):
    """A binary volume with a solid central cube (for contour actors)."""
    data = np.zeros((n, n, n), dtype=np.float32)
    data[5:15, 5:15, 5:15] = 1.0
    return data


def _label_volume(n=20):
    """A labeled volume with one nonzero region."""
    data = np.zeros((n, n, n), dtype=int)
    data[5:15, 5:15, 5:15] = 1
    return data


def _peak_dirs(n=8):
    """A (X, Y, Z, 3, 3) field of unit peak directions."""
    dirs = np.zeros((n, n, n, 3, 3), dtype=np.float32)
    dirs[..., 0, :] = (1.0, 0.0, 0.0)
    dirs[..., 1, :] = (0.0, 1.0, 0.0)
    dirs[..., 2, :] = (0.0, 0.0, 1.0)
    return dirs


def _vector_field(n=8):
    """A (X, Y, Z, 3) field of unit vectors."""
    field = np.zeros((n, n, n, 3), dtype=np.float32)
    field[...] = (1.0, 0.0, 0.0)
    return field


def _sph_coeffs(n=4, ncoeff=15):
    """SH coefficients with only the l=0 term -> isotropic visible glyphs."""
    coeffs = np.zeros((n, n, n, ncoeff), dtype=np.float32)
    coeffs[..., 0] = 1.0
    return coeffs


def _triangle_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return verts, faces


# --- The full set of renderable actors --------------------------------------
# Each entry is (id, factory). The factory builds a fresh actor on every call.

ACTOR_FACTORIES = {
    # Planar / polyhedron / curved mesh primitives (centers only).
    "square": lambda: actor.square(CENTERS),
    "triangle": lambda: actor.triangle(CENTERS),
    "star": lambda: actor.star(CENTERS),
    "disk": lambda: actor.disk(CENTERS),
    "ring": lambda: actor.ring(CENTERS),
    "box": lambda: actor.box(CENTERS),
    "tetrahedron": lambda: actor.tetrahedron(CENTERS),
    "icosahedron": lambda: actor.icosahedron(CENTERS),
    "triangularprism": lambda: actor.triangularprism(CENTERS),
    "pentagonalprism": lambda: actor.pentagonalprism(CENTERS),
    "octagonalprism": lambda: actor.octagonalprism(CENTERS),
    "rhombicuboctahedron": lambda: actor.rhombicuboctahedron(CENTERS),
    "frustum": lambda: actor.frustum(CENTERS),
    "superquadric": lambda: actor.superquadric(CENTERS),
    "cylinder": lambda: actor.cylinder(CENTERS),
    "cone": lambda: actor.cone(CENTERS),
    "arrow": lambda: actor.arrow(CENTERS),
    "sphere": lambda: actor.sphere(CENTERS),
    "ellipsoid": lambda: actor.ellipsoid(CENTERS),
    # Points / billboards.
    "point": lambda: actor.point(CENTERS),
    "marker": lambda: actor.marker(CENTERS),
    "billboard": lambda: actor.billboard(CENTERS),
    "billboard_sphere": lambda: actor.billboard_sphere(CENTERS),
    # Axes (no required args).
    "axes": lambda: actor.axes(),
    # Text / image.
    "text": lambda: actor.text("FURY"),
    "image": lambda: actor.image(_gradient_image()),
    # Line / streamline family.
    "line": lambda: actor.line(LINES),
    "streamlines": lambda: actor.streamlines(LINES),
    "streamtube": lambda: actor.streamtube(LINES),
    "line_projection": lambda: actor.line_projection(LINES),
    # Surface.
    "surface": lambda: actor.surface(*_triangle_mesh()),
    # Volume / field / glyph slicers (return single actors or groups; both are
    # hidden by setting ``.visible`` on the returned object).
    "data_slicer": lambda: actor.data_slicer(_scalar_volume()),
    "volume_slicer": lambda: actor.volume_slicer(_scalar_volume()),
    "peaks_slicer": lambda: actor.peaks_slicer(_peak_dirs()),
    "vector_field": lambda: actor.vector_field(_vector_field()),
    "vector_field_slicer": lambda: actor.vector_field_slicer(_vector_field()),
    "sph_glyph": lambda: actor.sph_glyph(_sph_coeffs()),
    # Contours (need a solid block to produce geometry).
    "contour_from_volume": lambda: actor.contour_from_volume(_roi_volume()),
    "contour_from_roi": lambda: actor.contour_from_roi(_roi_volume()),
    "contour_from_label": lambda: actor.contour_from_label(_label_volume()),
}


@pytest.mark.parametrize("name", list(ACTOR_FACTORIES))
def test_actor_visibility(name):
    """Every actor renders when visible and renders nothing when hidden."""
    obj = ACTOR_FACTORIES[name]()
    assert_visibility(obj, toggle=lambda v: setattr(obj, "visible", v))
