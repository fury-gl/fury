"""Shared helpers for actor module tests."""

from PIL import Image
import numpy as np
import numpy.testing as npt

from fury import actor, window

# --- Shared real inputs for visibility/snapshot tests -----------------------

CENTERS = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
LINES = [np.array([[0, 0, 0], [1, 1, 1], [2, 0, 0]], dtype=np.float32)]


def gradient_image(n=64):
    """A non-uniform 2D image so the rendered frame is never flat."""
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.outer(ramp, ramp)


def scalar_volume(n=32):
    """A non-uniform 3D scalar volume (deterministic)."""
    x, y, z = np.mgrid[0:n, 0:n, 0:n]
    return ((x + y + z) % 17).astype(np.float32)


def roi_volume(n=20):
    """A binary volume with a solid central cube (for contour actors)."""
    data = np.zeros((n, n, n), dtype=np.float32)
    data[5:15, 5:15, 5:15] = 1.0
    return data


def label_volume(n=20):
    """A labeled volume with one nonzero region."""
    data = np.zeros((n, n, n), dtype=int)
    data[5:15, 5:15, 5:15] = 1
    return data


def peak_dirs(n=8):
    """A (X, Y, Z, 3, 3) field of unit peak directions."""
    dirs = np.zeros((n, n, n, 3, 3), dtype=np.float32)
    dirs[..., 0, :] = (1.0, 0.0, 0.0)
    dirs[..., 1, :] = (0.0, 1.0, 0.0)
    dirs[..., 2, :] = (0.0, 0.0, 1.0)
    return dirs


def vector_field_data(n=8):
    """A (X, Y, Z, 3) field of unit vectors."""
    field = np.zeros((n, n, n, 3), dtype=np.float32)
    field[...] = (1.0, 0.0, 0.0)
    return field


def sph_coeffs(n=4, ncoeff=15):
    """SH coefficients with only the l=0 term -> isotropic visible glyphs."""
    coeffs = np.zeros((n, n, n, ncoeff), dtype=np.float32)
    coeffs[..., 0] = 1.0
    return coeffs


def triangle_mesh():
    """A single-triangle (vertices, faces) pair for surface actors."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return verts, faces


# The full set of renderable actors keyed by id. Each value is a factory that
# builds a fresh actor (with real inputs) on every call.
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
    "image": lambda: actor.image(gradient_image()),
    # Line / streamline family.
    "line": lambda: actor.line(LINES),
    "streamlines": lambda: actor.streamlines(LINES),
    "streamtube": lambda: actor.streamtube(LINES),
    "line_projection": lambda: actor.line_projection(LINES),
    # Surface.
    "surface": lambda: actor.surface(*triangle_mesh()),
    # Volume / field / glyph slicers (return single actors or groups; both are
    # hidden by setting ``.visible`` on the returned object).
    "data_slicer": lambda: actor.data_slicer(scalar_volume()),
    "volume_slicer": lambda: actor.volume_slicer(scalar_volume()),
    "peaks_slicer": lambda: actor.peaks_slicer(peak_dirs()),
    "vector_field": lambda: actor.vector_field(vector_field_data()),
    "vector_field_slicer": lambda: actor.vector_field_slicer(vector_field_data()),
    "sph_glyph": lambda: actor.sph_glyph(sph_coeffs()),
    # Contours (need a solid block to produce geometry).
    "contour_from_volume": lambda: actor.contour_from_volume(roi_volume()),
    "contour_from_roi": lambda: actor.contour_from_roi(roi_volume()),
    "contour_from_label": lambda: actor.contour_from_label(label_volume()),
}


def random_png(width, height):
    """
    Generates a random RGB PNG image.

    Parameters
    ----------
    width : int
        Width of the image in pixels.
    height : int
        Height of the image in pixels.

    Returns
    -------
    Image
        The generated image.
    """
    image = Image.new("RGB", (width, height))
    pixels = image.load()

    for x in range(width):
        for y in range(height):
            r = np.random.randint(0, 255)
            g = np.random.randint(0, 255)
            b = np.random.randint(0, 255)
            pixels[x, y] = (r, g, b)

    return image


def validate_actors(actor_type="actor_name", prim_count=1, **kwargs):
    scene = window.Scene()
    typ_actor = getattr(actor, actor_type)
    get_actor = typ_actor(**kwargs)
    scene.add(get_actor)

    centers = kwargs.get("centers", None)

    if centers is not None:
        npt.assert_array_equal(get_actor.local.position, centers[0])

        mean_vertex = np.round(np.mean(get_actor.geometry.positions.view, axis=0))
        npt.assert_array_almost_equal(mean_vertex, centers[0])

    assert get_actor.prim_count == prim_count

    if actor_type == "line":
        return

    # Default material: the actor renders and red dominates the image.
    arr = window.snapshot(scene=scene, fname=None, return_array=True)
    report = window.analyze_snapshot(arr, find_objects=True)
    assert report.objects >= 1

    mean_r, mean_g, mean_b, _mean_a = np.mean(arr.reshape(-1, arr.shape[2]), axis=0)
    assert mean_r > mean_b and mean_r > mean_g
    assert 0 < mean_r < 255

    # Hidden: nothing renders.
    get_actor.visible = False
    report = window.analyze_snapshot(
        window.snapshot(scene=scene, fname=None, return_array=True), find_objects=True
    )
    assert report.objects == 0
    scene.remove(get_actor)

    # Basic (flat) material: exact red is present, then absent once hidden.
    typ_actor_1 = getattr(actor, actor_type)
    get_actor_1 = typ_actor_1(**{**kwargs, "material": "basic"})
    scene.add(get_actor_1)

    report = window.analyze_snapshot(
        window.snapshot(scene=scene, fname=None, return_array=True),
        colors=(255, 0, 0),
        find_objects=True,
    )
    assert report.objects >= 1
    assert report.colors_found == [True]

    get_actor_1.visible = False
    report = window.analyze_snapshot(
        window.snapshot(scene=scene, fname=None, return_array=True),
        colors=(255, 0, 0),
        find_objects=True,
    )
    assert report.objects == 0
    assert report.colors_found == [False]
    scene.remove(get_actor_1)
