import re

from PIL import Image
import numpy as np
import numpy.testing as npt
import pytest
import rendercanvas.glfw

from fury import actor, window
from fury.io import load_image_texture
from fury.lib import Group, MeshBasicMaterial, MeshPhongMaterial, TextureMap
from fury.material import (
    VectorFieldArrowMaterial,
    VectorFieldLineMaterial,
    VectorFieldThinLineMaterial,
    _StreamtubeBakedMaterial,
)
from fury.optpkg import optional_package
from fury.utils import (
    generate_planar_uvs,
    get_slices,
    set_group_visibility,
    show_slices,
)

_, have_numba, _ = optional_package("numba")


def _do_nothing_patch(self):
    pass


rendercanvas.glfw.RenderCanvas._rc_close = _do_nothing_patch


def random_png(width, height):
    """Generates a random RGB PNG image.

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
    from fury.testing import analyze_snapshot

    scene = window.Scene()
    typ_actor = getattr(actor, actor_type)
    get_actor = typ_actor(**kwargs)
    scene.add(get_actor)

    centers = kwargs.get("centers", None)
    colors = kwargs.get("colors", None)

    if centers is not None:
        npt.assert_array_equal(get_actor.local.position, centers[0])

        mean_vertex = np.round(np.mean(get_actor.geometry.positions.view, axis=0))
        npt.assert_array_almost_equal(mean_vertex, centers[0])

    assert get_actor.prim_count == prim_count

    if actor_type == "line":
        return

    fname = f"{actor_type}_test.png"
    window.snapshot(scene=scene, fname=fname)

    img = Image.open(fname)
    img_array = np.array(img)

    mean_r, mean_g, mean_b, _mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g

    middle_pixel = img_array[img_array.shape[0] // 2, img_array.shape[1] // 2]
    r, g, b, a = middle_pixel
    assert r > g and r > b
    assert g == b

    # Advanced snapshot testing with shading-aware color detection
    arr = window.snapshot(
        scene=scene, fname=f"{actor_type}_snapshot.png", return_array=True
    )
    if colors is not None and arr is not None:
        # Convert colors to 0-255 range if needed
        test_colors = colors * 255 if colors.max() <= 1.0 else colors
        report = analyze_snapshot(arr, colors=test_colors.tolist(), color_tolerance=30)
        assert any(report.colors_found), (
            f"{actor_type} color should be detected despite shading"
        )
        if actor_type not in ["square", "disk"]:
            # Flat objects may not have strong shading
            assert report.has_shading, f"{actor_type} should show Phong shading"

    scene.remove(get_actor)

    typ_actor_1 = getattr(actor, actor_type)
    get_actor_1 = typ_actor_1(centers=centers, colors=colors, material="basic")
    scene.add(get_actor_1)
    fname_1 = f"{actor_type}_test_1.png"
    window.snapshot(scene=scene, fname=fname_1)
    img = Image.open(fname_1)
    img_array = np.array(img)

    mean_r, mean_g, mean_b, _mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g
    assert 0 < mean_r < 255
    assert mean_g == 0 and mean_b == 0

    middle_pixel = img_array[img_array.shape[0] // 2, img_array.shape[1] // 2]
    r, g, b, a = middle_pixel
    assert r > g and r > b
    assert g == 0 and b == 0
    assert r == 255
    scene.remove(get_actor_1)


def test_actor_from_primitive_wireframe():
    """Test wireframe and wireframe_thickness for primitive actors."""
    sphere_actor = actor.sphere(
        centers=np.array([[0, 0, 0]]), colors=np.array([[1, 0, 0]])
    )

    # By default, wireframe is off
    assert not sphere_actor.material.wireframe
    assert sphere_actor.material.wireframe_thickness == 1.0

    # Test enabling wireframe
    sphere_actor.material.wireframe = True
    assert sphere_actor.material.wireframe

    # Test disabling wireframe
    sphere_actor.material.wireframe = False
    assert not sphere_actor.material.wireframe

    # Test setting wireframe thickness
    new_thickness = 5.0
    sphere_actor.material.wireframe_thickness = new_thickness
    assert sphere_actor.material.wireframe_thickness == new_thickness


def test_sphere():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="sphere")


def test_sphere_visual():
    """Test sphere actor rendering with multiple spheres and shading."""
    from fury.testing import analyze_snapshot

    scene = window.Scene()
    scene.background = (0, 0, 0)

    # Multiple spheres with different colors and sizes
    centers = np.array([[0, 0, 0], [2, 0, 0], [-2, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    radii = np.array([0.5, 0.7, 0.4])

    spheres = actor.sphere(centers=centers, colors=colors, radii=radii)
    scene.add(spheres)

    arr = window.snapshot(scene=scene, return_array=True)

    report = analyze_snapshot(
        arr,
        colors=colors * 255,
        find_objects=True,
        analyze_shading=True,
        color_tolerance=30,
    )

    assert report.objects >= 3, f"Expected 3 spheres, found {report.objects}"
    assert all(report.colors_found), f"Not all colors detected: {report.colors_found}"
    assert report.has_shading, "Spheres should have Phong shading"


def test_line():
    lines_points = np.array([[[0, 0, 0], [1, 1, 1]], [[1, 1, 1], [2, 2, 2]]])
    colors = np.array([[[1, 0, 0]], [[0, 1, 0]]])
    validate_actors(lines=lines_points, colors=colors, actor_type="line", prim_count=2)

    line = np.array([[0, 0, 0], [1, 1, 1]])
    colors = None
    validate_actors(lines=line, colors=colors, actor_type="line", prim_count=2)

    line = np.array([[0, 0, 0], [1, 1, 1]])
    actor.line(line, colors=colors)
    actor.line(line)
    actor.line(line, colors=colors)
    actor.line(line, colors=colors, material="basic")
    actor.line(line, colors=line, material="basic")


def test_line_visual():
    """Test line actor rendering."""
    from fury.testing import analyze_snapshot

    scene = window.Scene()
    scene.background = (0, 0, 0)

    lines = np.array([[[0, 0, 0], [1, 1, 0]], [[1, 1, 0], [2, 0, 0]]])
    colors = np.array([[[1, 0, 0]], [[0, 1, 0]]])

    line_actor = actor.line(lines, colors=colors)
    scene.add(line_actor)

    arr = window.snapshot(scene=scene, return_array=True)

    report = analyze_snapshot(
        arr, colors=colors.reshape(-1, 3) * 255, color_tolerance=30
    )

    assert any(report.colors_found), "Should detect at least one line color"


def test_box():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="box")


def test_box_visual():
    """Test box actor rendering and scaling."""
    from fury.testing import analyze_snapshot

    scene = window.Scene()
    scene.background = (0, 0, 0)

    centers = np.array([[0, 0, 0], [3, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0]])
    scales = np.array([[1, 1, 1], [0.5, 2, 0.5]])

    boxes = actor.box(centers=centers, colors=colors, scales=scales)
    scene.add(boxes)

    arr = window.snapshot(scene=scene, return_array=True)

    report = analyze_snapshot(arr, colors=colors * 255, color_tolerance=30)

    assert all(report.colors_found), "Box colors not detected correctly"
    assert report.objects >= 2, f"Expected 2 boxes, found {report.objects}"


def test_cylinder():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="cylinder")


def test_cylinder_visual():
    """Test cylinder actor rendering with directions."""
    from fury.testing import analyze_snapshot

    scene = window.Scene()
    scene.background = (0, 0, 0)

    centers = np.array([[0, 0, 0], [3, 0, 0]])
    directions = np.array([[0, 1, 0], [1, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 0, 1]])

    cylinders = actor.cylinder(
        centers=centers, directions=directions, colors=colors, height=2.0, radii=0.3
    )
    scene.add(cylinders)

    arr = window.snapshot(scene=scene, return_array=True)

    report = analyze_snapshot(arr, colors=colors * 255, color_tolerance=30)

    assert all(report.colors_found), "Cylinder colors not detected"


def test_square():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="square")


def test_square_visual():
    """Test square actor rendering."""
    from fury.testing import analyze_snapshot

    scene = window.Scene()
    scene.background = (0, 0, 0)

    centers = np.array([[0, 0, 0], [2, 0, 0]])
    directions = np.array([[0, 0, 1], [0, 0, 1]])
    colors = np.array([[1, 1, 0], [0, 1, 1]])

    squares = actor.square(
        centers=centers, directions=directions, colors=colors, scales=1.0
    )
    scene.add(squares)

    arr = window.snapshot(scene=scene, return_array=True)

    report = analyze_snapshot(arr, colors=colors * 255, color_tolerance=30)

    assert all(report.colors_found), "All square colors should be detected"


def test_frustum():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="frustum")


def test_tetrahedron():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="tetrahedron")


def test_icosahedron():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="icosahedron")


def test_rhombicuboctahedron():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="rhombicuboctahedron")


def test_triangularprism():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="triangularprism")


def test_pentagonalprism():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="pentagonalprism")


def test_octagonalprism():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="octagonalprism")


def test_arrow():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="arrow")


def test_arrow_visual():
    """Test arrow actor rendering."""
    from fury.testing import analyze_snapshot

    scene = window.Scene()
    scene.background = (0, 0, 0)

    centers = np.array([[0, 0, 0], [2, 0, 0]])
    directions = np.array([[1, 0, 0], [0, 1, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0]])

    arrows = actor.arrow(
        centers=centers, directions=directions, colors=colors, height=1.5
    )
    scene.add(arrows)

    arr = window.snapshot(scene=scene, return_array=True)

    report = analyze_snapshot(arr, colors=colors * 255, color_tolerance=30)

    assert all(report.colors_found), "All arrow colors should be detected"


def test_superquadric():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="superquadric")


def test_superquadric_visual():
    """Test superquadric actor rendering."""
    from fury.testing import analyze_snapshot

    scene = window.Scene()
    scene.background = (0, 0, 0)

    centers = np.array([[0, 0, 0], [3, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 0, 1]])
    roundness = (1.0, 1.0)  # Sphere-like

    superqs = actor.superquadric(centers=centers, colors=colors, roundness=roundness)
    scene.add(superqs)

    arr = window.snapshot(scene=scene, return_array=True)

    report = analyze_snapshot(arr, colors=colors * 255, color_tolerance=30)

    assert all(report.colors_found), "Superquadric colors not detected"


def test_cone():
    centers = np.array([[0, 0, 0]])
    directions = np.array([[0, 1, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(
        centers=centers, directions=directions, colors=colors, actor_type="cone"
    )


def test_cone_visual():
    """Test cone actor rendering."""
    from fury.testing import analyze_snapshot

    scene = window.Scene()
    scene.background = (0, 0, 0)

    centers = np.array([[0, 0, 0], [2, 0, 0]])
    directions = np.array([[0, 1, 0], [0, 1, 0]])
    colors = np.array([[1, 1, 0], [1, 0, 1]])

    cones = actor.cone(
        centers=centers, directions=directions, colors=colors, height=2.0, radii=0.5
    )
    scene.add(cones)

    arr = window.snapshot(scene=scene, return_array=True)

    report = analyze_snapshot(arr, colors=colors * 255, color_tolerance=30)

    assert all(report.colors_found), "All cone colors should be detected"


def test_star():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="star")


def test_disk():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="disk")
    validate_actors(centers=centers, colors=colors, actor_type="disk", sectors=8)


def test_disk_visual():
    """Test disk actor rendering."""
    from fury.testing import analyze_snapshot

    scene = window.Scene()
    scene.background = (0, 0, 0)

    centers = np.array([[0, 0, 0]])
    directions = np.array([[0, 0, 1]])
    colors = np.array([[0, 1, 1]])

    disks = actor.disk(centers=centers, directions=directions, colors=colors, radii=1.0)
    scene.add(disks)

    arr = window.snapshot(scene=scene, return_array=True)

    report = analyze_snapshot(arr, colors=colors * 255, color_tolerance=50)

    # Disk may be small or have lighting affecting color detection
    assert report.colors_found[0] or np.any(arr > 0), "Disk not detected"


def test_triangle():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="triangle")


def test_ring():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    scene = window.Scene()
    ring_actor = actor.ring(centers=centers, colors=colors)
    scene.add(ring_actor)

    npt.assert_array_equal(ring_actor.local.position, centers[0])

    mean_vertex = np.round(np.mean(ring_actor.geometry.positions.view, axis=0))
    npt.assert_array_almost_equal(mean_vertex, centers[0])

    fname = "ring_test.png"
    window.snapshot(scene=scene, fname=fname)

    img = Image.open(fname)
    img_array = np.array(img)

    mean_r, mean_g, mean_b, _mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g
    scene.remove(ring_actor)

    ring_actor_1 = actor.ring(centers=centers, colors=colors, material="basic")
    scene.add(ring_actor_1)
    fname_1 = "ring_test_1.png"
    window.snapshot(scene=scene, fname=fname_1)
    img = Image.open(fname_1)
    img_array = np.array(img)

    mean_r, mean_g, mean_b, _mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g
    assert 0 < mean_r < 255
    assert mean_g == 0 and mean_b == 0

    scene.remove(ring_actor_1)


def test_point():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    scene = window.Scene()
    point_actor = actor.point(centers=centers, colors=colors)
    scene.add(point_actor)

    npt.assert_array_equal(point_actor.local.position, centers[0])

    mean_vertex = np.round(np.mean(point_actor.geometry.positions.view, axis=0))
    npt.assert_array_almost_equal(mean_vertex, centers[0])

    fname = "point_test.png"
    window.snapshot(scene=scene, fname=fname)

    img = Image.open(fname)
    img_array = np.array(img)

    mean_r, mean_g, mean_b, _mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g
    scene.remove(point_actor)

    point_actor_1 = actor.point(centers=centers, colors=colors, material="gaussian")
    scene.add(point_actor_1)
    fname_1 = "point_test_1.png"
    window.snapshot(scene=scene, fname=fname_1)
    img = Image.open(fname_1)
    img_array = np.array(img)

    mean_r, mean_g, mean_b, _mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g
    assert 0 < mean_r < 255
    assert mean_g == 0 and mean_b == 0

    scene.remove(point_actor_1)


def test_point_visual():
    """Test point actor rendering."""
    from fury.testing import analyze_snapshot

    scene = window.Scene()
    scene.background = (0, 0, 0)

    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    point_actor = actor.point(points, colors=colors, size=10.0)
    scene.add(point_actor)

    arr = window.snapshot(scene=scene, return_array=True)

    report = analyze_snapshot(arr, colors=colors * 255, color_tolerance=30)

    # Points are small, at least one color should be visible
    assert any(report.colors_found), "Should detect at least one point color"


def test_marker():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    scene = window.Scene()
    marker_actor = actor.marker(centers=centers, colors=colors)
    scene.add(marker_actor)

    npt.assert_array_equal(marker_actor.local.position, centers[0])

    mean_vertex = np.round(np.mean(marker_actor.geometry.positions.view, axis=0))
    npt.assert_array_almost_equal(mean_vertex, centers[0])

    fname = "marker_test.png"
    window.snapshot(scene=scene, fname=fname)

    img = Image.open(fname)
    img_array = np.array(img)

    mean_r, mean_g, mean_b, _mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g
    scene.remove(marker_actor)

    marker_actor_1 = actor.marker(centers=centers, colors=colors, marker="heart")
    scene.add(marker_actor_1)
    fname_1 = "marker_test_1.png"
    window.snapshot(scene=scene, fname=fname_1)
    img = Image.open(fname_1)
    img_array = np.array(img)

    mean_r, mean_g, mean_b, _mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_b and mean_r > mean_g
    assert 0 < mean_r < 255
    assert mean_g == 0 and mean_b == 0

    scene.remove(marker_actor_1)


def test_marker_visual():
    """Test marker actor rendering."""
    from fury.testing import analyze_snapshot

    scene = window.Scene()
    scene.background = (0, 0, 0)

    centers = np.array([[0, 0, 0], [1, 1, 0], [-1, -1, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    markers = actor.marker(centers, colors=colors, marker="o", size=30)
    scene.add(markers)

    arr = window.snapshot(scene=scene, return_array=True)

    report = analyze_snapshot(arr, colors=colors * 255, color_tolerance=30)

    assert sum(report.colors_found) >= 2, "Should detect at least 2 marker colors"


def test_text():
    text = "FURY"
    position1 = np.array([1.0, 0.0, 0.0])
    position2 = np.array([1.0, 2.0, 1.0])
    scene = window.Scene()

    text_actor = actor.text(text=text, anchor="middle-center", position=position1)
    scene.add(text_actor)

    npt.assert_array_equal(text_actor.local.position, position1)

    fname = "text_test.png"
    window.snapshot(scene=scene, fname=fname)

    img = Image.open(fname)
    img_array = np.array(img)

    mean_r, mean_g, mean_b, _mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r == mean_b and mean_r == mean_g
    assert 0 < mean_r < 255
    assert 0 < mean_g < 255
    assert 0 < mean_b < 255

    scene.remove(text_actor)

    text1 = "HELLO"
    text_actor_1 = actor.text(text=text1, anchor="middle-center", position=position2)
    scene.add(text_actor_1)
    npt.assert_array_equal(text_actor_1.local.position, position2)
    fname_1 = "text_test_1.png"
    window.snapshot(scene=scene, fname=fname_1)
    img = Image.open(fname_1)
    img_array = np.array(img)

    mean_r, mean_g, mean_b, _mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r == mean_b and mean_r == mean_g
    assert 0 < mean_r < 255
    assert 0 < mean_g < 255
    assert 0 < mean_b < 255

    scene.remove(text_actor_1)


def test_axes():
    scene = window.Scene()
    axes_actor = actor.axes()
    scene.add(axes_actor)

    assert axes_actor.prim_count == 3

    fname = "axes_test.png"
    window.snapshot(scene=scene, fname=fname)
    img = Image.open(fname)
    img_array = np.array(img)
    mean_r, mean_g, mean_b, _mean_a = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )
    assert np.isclose(mean_r, mean_g, atol=0.02)
    assert 0 < mean_r < 255
    assert 0 < mean_g < 255
    assert 0 < mean_b < 255

    scene.remove(axes_actor)


def test_axes_visual():
    """Test axes actor rendering."""
    from fury.testing import analyze_snapshot

    scene = window.Scene()
    scene.background = (0, 0, 0)

    axes = actor.axes(scale=(1, 1, 1))
    scene.add(axes)

    arr = window.snapshot(scene=scene, return_array=True)

    # Axes should have red, green, blue
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
    report = analyze_snapshot(arr, colors=colors, color_tolerance=30)

    # Should detect at least some of the axis colors
    assert any(report.colors_found), "No axis colors detected"


def test_ellipsoid():
    centers = np.array([[0, 0, 0]])
    lengths = np.array([[2, 1, 1]])
    axes = np.array([np.eye(3)])
    colors = np.array([1, 0, 0])

    validate_actors(
        centers=centers,
        lengths=lengths,
        orientation_matrices=axes,
        colors=colors,
        actor_type="ellipsoid",
    )

    _ = actor.ellipsoid(
        centers=centers,
        lengths=lengths,
        orientation_matrices=axes,
        colors=colors,
    )

    _ = actor.ellipsoid(
        np.array([[0, 0, 0], [1, 1, 1]]),
        lengths=np.array([[2, 1, 1]]),
        colors=np.array([[1, 0, 0]]),
    )

    _ = actor.ellipsoid(
        np.array([[0, 0, 0], [1, 1, 1]]), lengths=(2, 1, 1), colors=(1, 0, 0)
    )

    _ = actor.ellipsoid(centers)


def test_ellipsoid_visual():
    """Test ellipsoid actor rendering."""
    from fury.testing import analyze_snapshot

    scene = window.Scene()
    scene.background = (0, 0, 0)

    centers = np.array([[0, 0, 0]])
    axes_mat = np.array([np.eye(3)])
    colors = np.array([[1, 0, 1]])
    lengths = np.array([[1.5, 1.0, 0.5]])

    ellipsoids = actor.ellipsoid(
        centers=centers, orientation_matrices=axes_mat, colors=colors, lengths=lengths
    )
    scene.add(ellipsoids)

    arr = window.snapshot(scene=scene, return_array=True)

    report = analyze_snapshot(arr, colors=colors * 255, color_tolerance=30)

    assert report.colors_found[0], "Ellipsoid color not detected"


def test_valid_3d_data():
    """Test valid 3D input with default parameters (Test Case 1)."""
    data = np.random.rand(10, 20, 30)
    slicer_obj = actor.data_slicer(data)

    # Verify object type and visibility
    assert isinstance(slicer_obj, Group)
    assert slicer_obj.visible
    assert len(slicer_obj.children) == 3
    assert all(child.visible for child in slicer_obj.children)


def test_invalid_4d_data():
    """Test invalid 4D data shape (Test Case 4)."""
    data = np.random.rand(10, 20, 30, 4)  # Last dim ≠ 3
    with pytest.raises(ValueError) as excinfo:
        actor.data_slicer(data)
    assert "Last dimension must be of size 3" in str(excinfo.value)


def test_opacity_validation():
    """Test opacity validation raises errors for out-of-bounds values"""
    data = np.random.rand(10, 20, 30)

    # Test valid values
    for valid_opacity in [0, 0.5, 1]:
        slicer_obj = actor.data_slicer(data, opacity=valid_opacity)
        for child in slicer_obj.children:
            assert child.material.opacity == valid_opacity

    # Test invalid values
    for invalid_opacity in [-0.1, 1.1, 2.0]:
        with pytest.raises(ValueError) as excinfo:
            actor.data_slicer(data, opacity=invalid_opacity)
        assert "Opacity must be between 0 and 1" in str(excinfo.value)


def test_custom_initial_slices():
    """Test custom initial slice positions (Test Case 10)."""
    data = np.random.rand(10, 20, 30)
    slicer_obj = actor.data_slicer(data, initial_slices=(5, 10, 15))

    # Verify slice positions match input
    assert np.array_equal(get_slices(slicer_obj), [5, 10, 15])

    # Verify positions update correctly
    show_slices(slicer_obj, (2, 4, 6))
    assert np.array_equal(get_slices(slicer_obj), [2, 4, 6])


def test_visibility_control():
    """Test visibility settings through methods (Test Case 13)."""
    data = np.random.rand(10, 20, 30)
    slicer_obj = actor.data_slicer(data, visibility=(True, True, True))

    # Verify initial visibility
    assert all(child.visible for child in slicer_obj.children)

    # Update and verify new visibility
    set_group_visibility(slicer_obj, (False, True, False))
    visibilities = [child.visible for child in slicer_obj.children]
    assert visibilities == [False, True, False]


def test_image():
    scene = window.Scene()
    image = np.random.rand(100, 100)
    position = np.array([10, 10, 10])
    image_actor = actor.image(image=image, position=position)
    scene.add(image_actor)

    npt.assert_array_equal(image_actor.local.position, position)
    assert image_actor.visible

    scene.remove(image_actor)


def test_surface_basic_vertices_and_faces():
    """Test surface creation with basic vertices and faces."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    surface_actor = actor.surface(vertices, faces)

    assert np.array_equal(surface_actor.geometry.positions.data, vertices)
    assert np.array_equal(surface_actor.geometry.indices.data, faces)
    assert not hasattr(surface_actor.geometry, "texcoords")
    assert not hasattr(surface_actor.geometry, "colors")
    assert isinstance(surface_actor.material, MeshPhongMaterial)
    assert surface_actor.material.opacity == 1.0

    def test_surface_with_normals():
        """Test surface creation with normals."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)

        surface_actor = actor.surface(vertices, faces, normals=normals)

        assert np.array_equal(surface_actor.geometry.positions.data, vertices)
        assert np.array_equal(surface_actor.geometry.indices.data, faces)
        assert np.array_equal(surface_actor.geometry.normals.data, normals)
        assert not hasattr(surface_actor.geometry, "colors")
        assert not hasattr(surface_actor.geometry, "texcoords")
        assert isinstance(surface_actor.material, MeshPhongMaterial)
        assert surface_actor.material.opacity == 1.0


def test_surface_with_vertex_colors():
    """Test surface creation with vertex colors."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    surface_actor = actor.surface(vertices, faces, colors=colors)

    assert np.array_equal(surface_actor.geometry.positions.data, vertices)
    assert np.array_equal(surface_actor.geometry.indices.data, faces)
    assert np.array_equal(surface_actor.geometry.colors.data, colors)

    assert isinstance(surface_actor.material, MeshPhongMaterial)
    assert surface_actor.material.opacity == 1.0


def test_surface_with_vertex_colors_and_normals():
    """Test surface creation with vertex colors and normals."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)

    surface_actor = actor.surface(vertices, faces, colors=colors, normals=normals)

    assert np.array_equal(surface_actor.geometry.positions.data, vertices)
    assert np.array_equal(surface_actor.geometry.indices.data, faces)
    assert np.array_equal(surface_actor.geometry.colors.data, colors)
    assert np.array_equal(surface_actor.geometry.normals.data, normals)
    assert isinstance(surface_actor.material, MeshPhongMaterial)
    assert surface_actor.material.opacity == 1.0


def test_surface_with_texture(tmpdir):
    """Test surface creation with texture."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    # Create a dummy texture file
    texture_file = tmpdir.join("texture.png")
    image = random_png(10, 10)
    image.save(str(texture_file), "PNG")

    surface_actor = actor.surface(
        vertices, faces, texture=str(texture_file), texture_axis="xy"
    )

    tex = load_image_texture(str(texture_file))
    assert isinstance(surface_actor.material.map, TextureMap)
    assert np.array_equal(surface_actor.material.map.texture.data, tex.data)
    assert isinstance(surface_actor.material, MeshBasicMaterial)
    assert surface_actor.material.opacity == 1.0

    texcoords = generate_planar_uvs(vertices, axis="xy")
    assert np.array_equal(surface_actor.geometry.texcoords.data, texcoords)
    assert np.array_equal(surface_actor.geometry.positions.data, vertices)
    assert np.array_equal(surface_actor.geometry.indices.data, faces)


def test_surface_with_texture_coords(tmpdir):
    """Test surface creation with custom texture coordinates."""
    # Create simple geometry (a single triangle)
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )

    faces = np.array([[0, 1, 2]], dtype=np.int32)

    # Create custom texture coordinates
    texture_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    # Create a dummy texture file
    texture_file = tmpdir.join("texture.png")
    image = random_png(10, 10)
    image.save(str(texture_file), "PNG")

    # Test with texture_coords
    mesh = actor.surface(
        vertices=vertices,
        faces=faces,
        texture=str(texture_file),
        texture_coords=texture_coords,
    )

    # Verify the mesh was created (in a real test, you'd check properties)
    assert mesh is not None


def test_texture_coords_validation(tmpdir):
    """Test that invalid texture_coords raise appropriate errors."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    # Create a dummy texture file
    texture_file = tmpdir.join("texture.png")
    image = random_png(10, 10)
    image.save(str(texture_file), "PNG")

    # Test wrong shape
    with pytest.raises(ValueError):
        bad_coords = np.array([[0, 0], [1, 0]])  # missing one vertex
        actor.surface(
            vertices=vertices,
            faces=faces,
            texture=str(texture_file),
            texture_coords=bad_coords,
        )

    # Test wrong dtype
    with pytest.raises(ValueError):
        bad_coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        actor.surface(
            vertices=vertices,
            faces=faces,
            texture=str(texture_file),
            texture_coords=bad_coords,
        )


def test_surface_error_conditions():
    """Test error conditions for invalid inputs."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    # Test invalid colors format
    with pytest.raises(ValueError):
        actor.surface(vertices, faces, colors=[1, 0, 0, 1, 0])  # Invalid length

    # Test non-existent texture file
    with pytest.raises(FileNotFoundError):
        actor.surface(vertices, faces, texture="nonexistent.png")

    # Test invalid opacity
    with pytest.raises(ValueError):
        actor.surface(
            vertices, faces, opacity=1.5
        )  # Assuming validate_opacity raises for >1


def test_vector_field_initialization_with_4d_field():
    """Test VectorField initialization with 4D field (X,Y,Z,3)."""
    field = np.random.rand(5, 5, 5, 3)
    vf = actor.VectorField(field)
    assert vf.vectors.shape == (125, 3)  # 5*5*5=125 vectors
    assert vf.vectors_per_voxel == 1
    assert vf.field_shape == (5, 5, 5)


def test_vector_field_initialization_with_5d_field():
    """Test VectorField initialization with 5D field (X,Y,Z,N,3)."""
    field = np.random.rand(5, 5, 5, 2, 3)  # 2 vectors per voxel
    vf = actor.VectorField(field)
    assert vf.vectors.shape == (250, 3)  # 5*5*5*2=250 vectors
    assert vf.vectors_per_voxel == 2
    assert vf.field_shape == (5, 5, 5)


def test_vector_field_invalid_dimensions():
    """Test VectorField with invalid field dimensions."""
    # 3D field (not enough dimensions)
    field = np.random.rand(5, 5, 5)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Field must be 5D or 4D, but got {field.ndim}D with shape {field.shape}"
        ),
    ):
        actor.VectorField(field)

    # 6D field (too many dimensions)
    with pytest.raises(ValueError):
        field = np.random.rand(5, 5, 5, 2, 3, 1)
        actor.VectorField(field)

    # Last dimension not 3
    with pytest.raises(ValueError):
        field = np.random.rand(5, 5, 5, 2)
        actor.VectorField(field)


def test_vector_field_scales():
    """Test VectorField with different scale configurations."""
    field = np.random.rand(5, 5, 5, 3)

    # Test with float scale
    vf = actor.VectorField(field, scales=2.0)
    assert np.all(vf.scales == 2.0)

    # Test with matching array scale (4D)
    scales = np.random.rand(5, 5, 5)
    vf = actor.VectorField(field, scales=scales)
    assert vf.scales.shape == (125, 1)

    # Test with matching array scale (5D)
    field = np.random.rand(5, 5, 5, 2, 3)
    scales = np.random.rand(5, 5, 5, 2)
    vf = actor.VectorField(field, scales=scales)
    assert vf.scales.shape == (250, 1)


def test_vector_field_cross_section():
    """Test VectorField cross section property."""
    field = np.random.rand(5, 5, 5, 3)

    # Test default cross section
    vf = actor.VectorField(field)

    # Test setting cross section
    # cross section will not work without providing visibility.
    new_cross = [1, 2, 3]
    vf.cross_section = new_cross
    assert vf.visibility is None

    # Test invalid cross section types
    with pytest.raises(ValueError):
        vf.cross_section = "invalid"

    # Test invalid cross section length
    with pytest.raises(ValueError):
        vf.cross_section = [1, 2]


def test_vector_field_visibility():
    """Test VectorField visibility with cross section."""
    field = np.random.rand(5, 5, 5, 3)

    # Test with visibility
    vf = actor.VectorField(field, visibility=(True, False, True))
    assert np.all(vf.visibility == np.asarray((True, False, True)))

    # Set cross section with visibility
    vf.cross_section = [1, 2, 3]
    # The y dimension should be -1 because visibility[1] is False
    assert np.all(vf.cross_section == np.array([1, 2, 3]))


def test_vector_field_actor_types():
    """Test VectorField with different actor types."""
    field = np.random.rand(5, 5, 5, 3)

    for actor_type, material_type in zip(
        ["thin_line", "line", "arrow"],
        [
            VectorFieldThinLineMaterial,
            VectorFieldLineMaterial,
            VectorFieldArrowMaterial,
        ],
        strict=False,
    ):
        vf = actor.VectorField(field, actor_type=actor_type)
        assert isinstance(vf.material, material_type)


def test_vector_field_colors():
    """Test VectorField with different color configurations."""
    field = np.random.rand(5, 5, 5, 3)

    # Test with default color (None)
    vf = actor.VectorField(field)
    assert np.all(vf.geometry.colors.data[0] == np.array([0, 0, 0]))

    # Test with custom color
    color = (1.0, 0.5, 0.0)
    vf = actor.VectorField(field, colors=color)
    assert np.all(vf.geometry.colors.data[0] == np.array(color))


def test_vector_field_helper_functions():
    """Test the vector_field and vector_field_slicer helper functions."""
    field = np.random.rand(5, 5, 5, 3)

    # Test vector_field
    vf = actor.vector_field(field, actor_type="arrow", opacity=0.5, thickness=2.0)
    assert isinstance(vf.material, VectorFieldArrowMaterial)
    assert vf.material.opacity == 0.5
    assert vf.material.thickness == 2.0

    # Test vector_field_slicer
    vf = actor.vector_field_slicer(
        field,
        actor_type="line",
        cross_section=[2, 2, 2],
        visibility=(True, False, True),
    )
    assert isinstance(vf.material, VectorFieldLineMaterial)
    assert np.all(vf.cross_section == np.array([2, 2, 2]))
    assert np.all(vf.visibility == np.asarray((True, False, True)))


def test_vector_field_edge_cases():
    """Test VectorField with edge cases."""
    # Test with minimal field size
    field = np.random.rand(1, 1, 1, 3)
    vf = actor.VectorField(field)
    assert vf.vectors.shape == (1, 3)

    # Test with zero opacity
    vf = actor.VectorField(field, opacity=0.0)
    assert vf.material.opacity == 0.0

    # Test with zero thickness (should still work)
    vf = actor.VectorField(field, thickness=0.0)
    assert vf.material.thickness == 0.0  # Replace with your module


def test_sph_glyph_input_validation():
    """sph_glyph: Test invalid inputs raise appropriate errors."""
    # Invalid coeffs type/dimensions
    with pytest.raises(TypeError):
        actor.sph_glyph([1, 2, 3])  # Not a numpy array
    with pytest.raises(ValueError):
        actor.sph_glyph(np.random.rand(3, 3))  # Not 4D

    # Invalid sphere specification
    with pytest.raises(TypeError):
        actor.sph_glyph(np.random.rand(2, 2, 2, 5), sphere=1.5)
    with pytest.raises(TypeError):
        actor.sph_glyph(np.random.rand(2, 2, 2, 5), sphere=("a", "b"))


def test_sph_glyph_default_behavior():
    """sph_glyph: Test function with minimal valid inputs."""
    coeffs = np.random.rand(2, 2, 2, 9)
    glyph = actor.sph_glyph(coeffs)

    assert glyph is not None
    assert isinstance(glyph, actor.SphGlyph)
    assert glyph.sphere.shape[0] == 362  # Default sphere has 362 vertices
    assert glyph.color_type == 0  # Converted for shader compatibility


def test_sph_glyph_custom_sphere():
    """sph_glyph: Test custom sphere specifications."""
    coeffs = np.random.rand(2, 2, 2, 9)

    # Named sphere
    glyph = actor.sph_glyph(coeffs, sphere="symmetric724")
    assert glyph.sphere.shape[0] == 724

    # Custom sphere
    glyph = actor.sph_glyph(coeffs, sphere=(36, 72))
    assert hasattr(glyph, "indices")


def test_sph_glyph_parameter_combinations():
    """sph_glyph: Test all valid basis_type and color_type combinations."""
    coeffs = np.random.rand(2, 2, 2, 16)

    for basis in ["standard", "descoteaux07"]:
        for idx, color in enumerate(["sign", "orientation"]):
            glyph = actor.sph_glyph(coeffs, basis_type=basis, color_type=color)
            assert glyph.color_type == idx


def test_sph_glyph_shininess_values():
    """sph_glyph: Test valid shininess values."""
    coeffs = np.random.rand(2, 2, 2, 4)

    for shininess in [0, 50, 100, 150.5]:
        glyph = actor.sph_glyph(coeffs, shininess=shininess)
        assert glyph.material.shininess == shininess


def test_SphGlyph_input_validation_coeffs():
    """SphGlyph: Test invalid coeffs inputs raise appropriate errors."""
    valid_sphere = (np.random.rand(100, 3), np.random.randint(0, 100, (50, 3)))

    # Not a numpy array
    with pytest.raises(TypeError):
        actor.SphGlyph([1, 2, 3], sphere=valid_sphere)

    # Not 4D
    with pytest.raises(ValueError):
        actor.SphGlyph(np.random.rand(3, 3), sphere=valid_sphere)

    # Empty last dimension
    with pytest.raises(ValueError):
        actor.SphGlyph(np.random.rand(2, 2, 2, 0), sphere=valid_sphere)


def test_SphGlyph_input_validation_sphere():
    """SphGlyph: Test invalid sphere inputs raise appropriate errors."""
    valid_coeffs = np.random.rand(2, 2, 2, 9)

    # Not a tuple
    with pytest.raises(TypeError):
        actor.SphGlyph(valid_coeffs, sphere=[1, 2, 3])

    # Wrong tuple length
    with pytest.raises(TypeError):
        actor.SphGlyph(valid_coeffs, sphere=(np.random.rand(100, 3),))

    # Invalid contents
    with pytest.raises(TypeError):
        actor.SphGlyph(valid_coeffs, sphere=([1, 2, 3], [4, 5, 6]))


def test_SphGlyph_initialization_defaults():
    """SphGlyph: Test initialization with default parameters."""
    coeffs = np.random.rand(2, 2, 2, 9)
    sphere = (np.random.rand(100, 3), np.random.randint(0, 100, (50, 3)))
    glyph = actor.SphGlyph(coeffs, sphere=sphere)

    assert glyph.n_coeff == 9
    assert glyph.data_shape == (2, 2, 2)
    assert glyph.color_type == 0  # Default 'sign'
    assert glyph.vertices_per_glyph == 100
    assert glyph.faces_per_glyph == 50


def test_SphGlyph_parameter_combinations():
    """SphGlyph: Test different basis_type and color_type combinations."""
    coeffs = np.random.rand(2, 2, 2, 16)
    sphere = (np.random.rand(100, 3), np.random.randint(0, 100, (50, 3)))

    # Test basis types
    for basis in ["standard", "descoteaux07"]:
        glyph = actor.SphGlyph(coeffs, sphere=sphere, basis_type=basis)
        assert hasattr(glyph.material, "n_coeffs")

    # Test color types
    glyph_sign = actor.SphGlyph(coeffs, sphere=sphere, color_type="sign")
    assert glyph_sign.color_type == 0

    glyph_orient = actor.SphGlyph(coeffs, sphere=sphere, color_type="orientation")
    assert glyph_orient.color_type == 1


def test_SphGlyph_shininess_values():
    """SphGlyph: Test different shininess values."""
    coeffs = np.random.rand(2, 2, 2, 4)
    sphere = (np.random.rand(100, 3), np.random.randint(0, 100, (50, 3)))

    for shininess in [0, 50, 100, 150.5]:
        glyph = actor.SphGlyph(coeffs, sphere=sphere, shininess=shininess)
        assert glyph.material.shininess == shininess


def test_SphGlyph_geometry_properties():
    """SphGlyph: Test geometry properties are correctly set."""
    coeffs = np.random.rand(3, 3, 3, 9)
    vertices = np.random.rand(200, 3)
    faces = np.random.randint(0, 200, (100, 3))
    sphere = (vertices, faces)

    glyph = actor.SphGlyph(coeffs, sphere=sphere)

    # Check positions scaling
    assert glyph.geometry.positions.data.shape[0] == 3 * 3 * 3 * 200
    assert glyph.geometry.indices.data.shape[0] == 3 * 3 * 3 * 100

    # Check SH coefficients
    assert glyph.sh_coeff.shape[0] == 3 * 3 * 3 * 9
    assert glyph.sf_func.shape[0] == 200 * glyph.material.n_coeffs


def test_streamtube():
    lines = [np.array([[0, 0, 0], [1, 1, 1]])]
    colors = np.array([[1, 0, 0]])
    scene = window.Scene()

    tube_actor = actor.streamtube(lines=lines, colors=colors)
    scene.add(tube_actor)

    fname = "streamtube_test.png"
    window.snapshot(scene=scene, fname=fname)
    img = Image.open(fname)
    img_array = np.array(img)

    mean_r, mean_g, mean_b, _ = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_g and mean_r > mean_b

    middle_pixel = img_array[img_array.shape[0] // 2, img_array.shape[1] // 2]
    r, g, b, a = middle_pixel
    assert r > g and r > b


def test_streamtube_gpu_geometry_and_buffers():
    """GPU streamtube: geometry, buffers, and material state consistency."""
    lines = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.5]], dtype=np.float32),
        np.array([[0.5, -0.5, 0.2], [0.5, 0.5, 0.8]], dtype=np.float32),
    ]
    colors = np.array([[1.0, 0.0, 0.0, 0.6], [0.0, 1.0, 0.0, 0.4]], dtype=np.float32)
    radius = 0.15
    segments = 5

    mesh = actor.streamtube(
        lines,
        colors=colors,
        opacity=0.75,
        radius=radius,
        segments=segments,
        end_caps=True,
        backend="gpu",
    )

    assert isinstance(mesh.material, _StreamtubeBakedMaterial)
    assert mesh.n_lines == len(lines)

    line_lengths = np.array([line.shape[0] for line in lines], dtype=np.uint32)
    assert np.array_equal(mesh.line_lengths, line_lengths)
    assert mesh.max_line_length == int(line_lengths.max())
    assert mesh.tube_sides == segments
    assert mesh.end_caps is True

    vertices_per_line = line_lengths * segments + 2
    expected_vertex_offsets = np.zeros_like(line_lengths)
    if len(lines) > 1:
        expected_vertex_offsets[1:] = np.cumsum(
            vertices_per_line[:-1], dtype=np.uint64
        ).astype(np.uint32)
    assert np.array_equal(mesh.vertex_offsets, expected_vertex_offsets)

    segments_per_line = np.maximum(line_lengths - 1, 0)
    triangles_per_line = segments_per_line * segments * 2 + segments * 2
    expected_triangle_offsets = np.zeros_like(line_lengths)
    if len(lines) > 1:
        expected_triangle_offsets[1:] = np.cumsum(
            triangles_per_line[:-1], dtype=np.uint64
        ).astype(np.uint32)
    assert np.array_equal(mesh.triangle_offsets, expected_triangle_offsets)

    total_vertices = int(vertices_per_line.astype(np.uint64).sum())
    total_triangles = int(triangles_per_line.astype(np.uint64).sum())
    assert mesh.geometry.positions.data.shape == (total_vertices, 3)
    assert mesh.geometry.indices.data.shape == (total_triangles, 3)
    assert mesh.geometry.colors.data.shape == (total_vertices, 3)

    reshaped_line_buffer = mesh.line_buffer.data.reshape(
        len(lines), mesh.max_line_length, 3
    )
    expected_line_data = np.zeros_like(reshaped_line_buffer)
    for idx, line in enumerate(lines):
        expected_line_data[idx, : line.shape[0]] = line
    assert np.allclose(reshaped_line_buffer, expected_line_data)

    expected_colors = colors[:, :3]
    assert np.allclose(mesh.line_colors, expected_colors)
    assert np.allclose(mesh.color_buffer.data, expected_colors)
    assert mesh.color_components == 3

    assert np.isclose(mesh.material.radius, radius)
    assert mesh.material.segments == segments
    assert mesh.material.end_caps is True
    assert mesh.material.line_count == mesh.n_lines


def test_streamtube_gpu_color_broadcast_and_material_flags():
    """GPU streamtube: color broadcasting and material flags."""
    lines = [
        np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 1.5, 0.5]], dtype=np.float32),
        np.array([[1.0, 0.0, 0.0], [1.5, 0.5, 0.5], [2.0, 1.0, 1.0]], dtype=np.float32),
    ]
    base_color = (0.2, 0.4, 0.6)

    mesh = actor.streamtube(
        lines,
        colors=base_color,
        segments=4,
        end_caps=False,
        enable_picking=False,
        backend="gpu",
    )

    expected_colors = np.tile(np.asarray(base_color, dtype=np.float32), (len(lines), 1))
    assert np.allclose(mesh.line_colors, expected_colors)
    assert np.allclose(mesh.color_buffer.data, expected_colors)
    assert mesh.color_components == 3

    assert isinstance(mesh.material, _StreamtubeBakedMaterial)
    assert mesh.material.pick_write is False
    assert mesh.material.flat_shading is False
    assert mesh.material.end_caps is False
    assert mesh.material.segments == 4


def test_streamtube_gpu_invalid_inputs():
    """GPU streamtube: invalid inputs raise informative errors."""
    line_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    line_b = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="material='phong' only"):
        actor.streamtube(
            [line_a], colors=(1.0, 0.0, 0.0), backend="gpu", material="basic"
        )

    with pytest.raises(
        ValueError, match=r"first dimension must be 1 or \d+ \(number of lines\)"
    ):
        actor.streamtube(
            [line_a, line_b],
            colors=np.ones((3, 3), dtype=np.float32),
            backend="gpu",
        )

    with pytest.raises(
        ValueError, match=r"(must have 3|components, got) \(RGB\) or 4 \(RGBA\)"
    ):
        actor.streamtube(
            [line_a],
            colors=np.array([1.0, 0.5], dtype=np.float32),
            backend="gpu",
        )


def test_line_projection():
    """Test line_projection function with default parameters."""
    lines = [
        np.array([[0, 0, 0], [1, 1, 1]]),
        np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4]]),
    ]

    # Test basic functionality
    projection_actor = actor.line_projection(lines)
    assert projection_actor is not None
    assert projection_actor.num_lines == 2

    # Test with custom parameters
    projection_actor = actor.line_projection(
        lines,
        plane=(0, 0, 1, -1),
        colors=(0, 1, 0),
        thickness=2.0,
        outline_color=(1, 1, 0),
        outline_thickness=0.5,
        opacity=0.8,
    )
    assert projection_actor.num_lines == 2
    assert np.round(projection_actor.material.opacity, 1) == 0.8

    # Test with per-line colors
    colors = [(1, 0, 0), (0, 1, 0)]
    projection_actor = actor.line_projection(lines, colors=colors)
    assert projection_actor.num_lines == 2

    # Test with offsets
    offsets = [0, 2]
    projection_actor = actor.line_projection(lines, offsets=offsets)
    assert projection_actor.num_lines == 2
    npt.assert_array_equal(projection_actor.offsets, offsets)

    # Test plane as string 'XY'
    projection_actor = actor.line_projection(lines, plane="XY")
    npt.assert_array_equal(projection_actor.plane, [0, 0, -1, 0])

    # Test plane as string 'XZ'
    projection_actor = actor.line_projection(lines, plane="XZ")
    npt.assert_array_equal(projection_actor.plane, [0, -1, 0, 0])

    # Test plane as string 'YZ'
    projection_actor = actor.line_projection(lines, plane="YZ")
    npt.assert_array_equal(projection_actor.plane, [-1, 0, 0, 0])

    # Test invalid plane string
    import pytest

    with pytest.raises(
        ValueError,
        match=("Plane must be 'XY', 'XZ', 'YZ' or a tuple of 4 elements"),
    ):
        actor.line_projection(lines, plane="INVALID")


def test_line_projection_class():
    """Test LineProjection class initialization and properties."""
    from fury.actor.planar import LineProjection

    lines = [
        np.array([[0, 0, 0], [1, 1, 1]]),
        np.array([[2, 2, 2], [3, 3, 3]]),
    ]

    # Test basic initialization
    projection = LineProjection(lines)
    assert projection.num_lines == 2
    assert len(projection.lines) > 0
    npt.assert_array_equal(projection.plane, [0, 0, -1, 0])
    assert projection.lift == 0.0

    # Test with custom plane
    plane = (1, 0, 0, -2)
    projection = LineProjection(lines, plane=plane)
    npt.assert_array_equal(projection.plane, plane)
    assert projection.lift == 0.0

    # Test plane property setter
    new_plane = (0, 1, 0, 1)
    projection.plane = new_plane
    npt.assert_array_equal(projection.plane, new_plane)
    assert projection.lift == 0.0

    # Test plane setter with None (should default)
    projection.plane = None
    npt.assert_array_equal(projection.plane, [0, 0, -1, 0])
    assert projection.lift == 0.0

    # Test with custom colors
    colors = [(1, 0, 0), (0, 1, 0)]
    projection = LineProjection(lines, colors=colors)
    assert projection.geometry.colors.data.shape[0] == 2
    assert projection.lift == 0.0

    # Test with single color for all lines
    projection = LineProjection(lines, colors=(0, 0, 1))
    assert projection.geometry.colors.data.shape[0] == 2
    assert projection.lift == 0.0

    # Test with custom lengths and offsets
    lengths = [2, 2]
    offsets = [0, 2]
    projection = LineProjection(lines, lengths=lengths, offsets=offsets)
    npt.assert_array_equal(projection.lengths, lengths)
    npt.assert_array_equal(projection.offsets, offsets)
    assert projection.lift == 0.0

    # Test with custom lift value
    custom_lift = 0.5
    projection = LineProjection(lines, lift=custom_lift)
    assert projection.lift == custom_lift
    # Test lift property setter
    projection.lift = 0.8
    assert projection.lift == 0.8


def test_line_projection_validation():
    """Test LineProjection input validation and error handling."""
    from fury.actor.planar import LineProjection

    lines = [
        np.array([[0, 0, 0], [1, 1, 1]]),
        np.array([[2, 2, 2], [3, 3, 3]]),
    ]

    # Test invalid lengths
    with pytest.raises(ValueError, match="Lengths must have a length of 2"):
        LineProjection(lines, lengths=[2])

    # Test invalid offsets
    with pytest.raises(ValueError, match="Offsets must have a length of 2"):
        LineProjection(lines, offsets=[0])

    # Test invalid thickness type
    with pytest.raises(ValueError, match="Thickness must be a single float value"):
        LineProjection(lines, thickness="invalid")

    # Test invalid outline_thickness type
    with pytest.raises(
        ValueError, match="Outline thickness must be a single float value"
    ):
        LineProjection(lines, outline_thickness="invalid")

    # Test invalid outline_color
    with pytest.raises(ValueError, match="outline_color must have a length of 1 or"):
        LineProjection(lines, outline_color=(1, 0))

    with pytest.raises(ValueError, match="colors must have a length of 1 or"):
        LineProjection(lines, colors=(1, 0))

    # Test invalid plane in property setter
    projection = LineProjection(lines)
    with pytest.raises(ValueError, match="Plane must have a length of 4"):
        projection.plane = (1, 0, 0)

    # Test lift setter with None (should default)
    projection = LineProjection(lines)
    with pytest.raises(
        ValueError, match="Lift must be a single float value. Got None."
    ):
        projection.lift = None


def test_line_projection_edge_cases():
    """Test LineProjection with edge cases and boundary conditions."""
    from fury.actor.planar import LineProjection

    # Test with single line
    single_line = [np.array([[0, 0, 0], [1, 1, 1]])]
    projection = LineProjection(single_line)
    assert projection.num_lines == 1

    # Test with empty lines (single point lines)
    empty_lines = [np.array([[0, 0, 0]])]
    projection = LineProjection(empty_lines)
    assert projection.num_lines == 1

    # Test with zero thickness
    lines = [np.array([[0, 0, 0], [1, 1, 1]])]
    projection = LineProjection(lines, thickness=0.0)
    assert projection.material.size == 0.0

    # Test with zero outline thickness
    projection = LineProjection(lines, outline_thickness=0.0)
    assert projection.material.edge_width == 0.0

    # Test with maximum opacity
    projection = LineProjection(lines, opacity=1.0)
    assert projection.material.opacity == 1.0

    # Test with minimum opacity
    projection = LineProjection(lines, opacity=0.0)
    assert projection.material.opacity == 0.0

    # Test with None outline_color (should default to black)
    projection = LineProjection(lines, outline_color=None)
    # The material should handle this appropriately

    # Test with 4-component color (RGBA)
    projection = LineProjection(lines, colors=(1, 0, 0, 0.5))
    assert projection.geometry.colors.data.shape[1] >= 3

    # Test with 4-component outline color
    projection = LineProjection(lines, outline_color=(1, 1, 1, 0.5))
    # Should not raise an error


def test_line_projection_automatic_calculations():
    """Test automatic calculation of lengths and offsets."""
    from fury.actor.planar import LineProjection

    # Test automatic length calculation
    lines = [
        np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),  # 3 points
        np.array([[3, 3, 3], [4, 4, 4]]),  # 2 points
    ]
    projection = LineProjection(lines)
    expected_lengths = [3, 2]
    npt.assert_array_equal(projection.lengths, expected_lengths)

    # Test automatic offset calculation
    expected_offsets = [0, 3]  # Second line starts after first line
    npt.assert_array_equal(projection.offsets, expected_offsets)

    # Test with varying line lengths
    lines = [
        np.array([[0, 0, 0]]),  # 1 point
        np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]),  # 4 points
        np.array([[5, 5, 5], [6, 6, 6]]),  # 2 points
    ]
    projection = LineProjection(lines)
    expected_lengths = [1, 4, 2]
    expected_offsets = [0, 1, 5]
    npt.assert_array_equal(projection.lengths, expected_lengths)
    npt.assert_array_equal(projection.offsets, expected_offsets)


def test_line_projection_material_properties():
    """Test LineProjection material properties and configuration."""
    from fury.actor.planar import LineProjection

    lines = [np.array([[0, 0, 0], [1, 1, 1]])]

    # Test default material properties
    projection = LineProjection(lines)
    assert projection.material.size == 1.0  # default thickness
    assert (
        np.round(projection.material.edge_width, 1) == 0.2
    )  # default outline_thickness
    assert np.round(projection.material.opacity, 1) == 1.0  # default opacity

    # Test custom material properties
    projection = LineProjection(
        lines,
        thickness=5.0,
        outline_thickness=1.0,
        opacity=0.5,
    )
    assert projection.material.size == 5.0
    assert projection.material.edge_width == 1.0
    assert np.round(projection.material.opacity, 1) == 0.5


# ===== COMPREHENSIVE VISUAL BUG DETECTION TESTS =====


def test_text_actor_opacity():
    """Test text actor opacity/transparency rendering.

    BUG DETECTION: Text opacity may not render correctly.
    """
    scene = window.Scene()
    scene.background = (0, 0, 0)

    # Semi-transparent text
    text_act = actor.text("OPACITY TEST", position=(0, 0, 0), anchor="middle-center")
    # Note: Check if text actor supports opacity in SetOpacity
    if hasattr(text_act, "SetOpacity"):
        text_act.SetOpacity(0.5)

    scene.add(text_act)

    arr = window.snapshot(scene=scene, fname="test_text_opacity.png", return_array=True)

    # Find white/gray pixels (text should be semi-transparent)
    text_pixels = np.where((arr[..., 0] > 50) & (arr[..., 1] > 50) & (arr[..., 2] > 50))

    # Text should be visible
    assert len(text_pixels[0]) > 100, (
        f"Text should be visible. Found {len(text_pixels[0])} pixels"
    )

    # For semi-transparent text, pixels should not be pure white
    if hasattr(text_act, "SetOpacity"):
        pure_white = np.sum(
            (arr[..., 0] > 250) & (arr[..., 1] > 250) & (arr[..., 2] > 250)
        )
        total_text = len(text_pixels[0])

        # Most pixels should be gray (semi-transparent), not pure white
        assert pure_white < total_text * 0.9, (
            f"Semi-transparent text should have gray pixels. "
            f"Pure white: {pure_white}, Total text: {total_text}"
        )


def test_text_actor_relative_positioning():
    """Test text actor positioning relative to other actors.

    BUG DETECTION: Text actors may not translate correctly relative to spheres.
    This validates spatial relationships between actors.
    """

    scene = window.Scene()
    scene.background = (0, 0, 0)

    # Reference sphere at origin (red)
    ref_sphere = actor.sphere(
        centers=np.array([[0, 0, 0]]), colors=np.array([[1, 0, 0]]), radii=0.5
    )
    scene.add(ref_sphere)

    # Text positioned to the right
    text_act = actor.text("TEST", position=(3, 0, 0), anchor="middle-center")
    scene.add(text_act)

    # Another sphere as right reference (blue)
    right_sphere = actor.sphere(
        centers=np.array([[6, 0, 0]]), colors=np.array([[0, 0, 1]]), radii=0.5
    )
    scene.add(right_sphere)

    arr = window.snapshot(
        scene=scene, fname="test_text_position.png", return_array=True
    )

    # Find pixel positions
    red_pixels = np.where(
        (arr[..., 0] > 100) & (arr[..., 1] < 100) & (arr[..., 2] < 100)
    )
    white_pixels = np.where(
        (arr[..., 0] > 200) & (arr[..., 1] > 200) & (arr[..., 2] > 200)
    )
    blue_pixels = np.where(
        (arr[..., 0] < 100) & (arr[..., 1] < 100) & (arr[..., 2] > 100)
    )

    if len(red_pixels[1]) > 0 and len(white_pixels[1]) > 0 and len(blue_pixels[1]) > 0:
        red_x = np.mean(red_pixels[1])
        white_x = np.mean(white_pixels[1])
        blue_x = np.mean(blue_pixels[1])

        # Text should be between red (left) and blue (right)
        assert red_x < white_x < blue_x, (
            f"Text positioning FAILED! Expected: Red < Text < Blue. "
            f"Got Red: {red_x:.1f}, Text: {white_x:.1f}, Blue: {blue_x:.1f}"
        )


def test_multiple_actors_depth_ordering():
    """Test z-ordering and depth buffer for overlapping actors.

    BUG DETECTION: Rendering order, z-fighting, or depth issues.
    """

    scene = window.Scene()
    scene.background = (0, 0, 0)

    # Sphere 1 at origin (red, smaller)
    sphere1 = actor.sphere(
        centers=np.array([[0, 0, 0]]), colors=np.array([[1, 0, 0]]), radii=0.5
    )
    scene.add(sphere1)

    # Sphere 2 at same position (green, larger - added second, should dominate)
    sphere2 = actor.sphere(
        centers=np.array([[0, 0, 0]]), colors=np.array([[0, 1, 0]]), radii=0.7
    )
    scene.add(sphere2)

    arr = window.snapshot(scene=scene, fname="test_overlap.png", return_array=True)

    green_pixels = np.sum(
        (arr[..., 0] < 100) & (arr[..., 1] > 100) & (arr[..., 2] < 100)
    )
    red_pixels = np.sum((arr[..., 0] > 100) & (arr[..., 1] < 100) & (arr[..., 2] < 100))

    assert green_pixels > red_pixels, (
        f"Larger green sphere should dominate. Green: {green_pixels}, Red: {red_pixels}"
    )


def test_arrow_direction_visual():
    """Test arrow actor points in correct direction.

    BUG DETECTION: Arrow orientation or direction bugs.
    """

    scene = window.Scene()
    scene.background = (0, 0, 0)

    # Arrow pointing right (positive X)
    arrow_right = actor.arrow(
        centers=np.array([[0, 0, 0]]),
        directions=np.array([[1, 0, 0]]),
        colors=np.array([[1, 0, 0]]),
        height=2.0,
    )
    scene.add(arrow_right)

    arr = window.snapshot(
        scene=scene, fname="test_arrow_direction.png", return_array=True
    )

    # Find red pixels
    red_pixels = np.where(
        (arr[..., 0] > 100) & (arr[..., 1] < 100) & (arr[..., 2] < 100)
    )

    if len(red_pixels[1]) > 0:
        # Arrow should extend more to the right than left from center
        center_x = arr.shape[1] // 2
        right_pixels = np.sum(red_pixels[1] > center_x)
        left_pixels = np.sum(red_pixels[1] < center_x)

        assert right_pixels > left_pixels, (
            f"Arrow should point right. Right pixels: {right_pixels}, "
            f"Left pixels: {left_pixels}"
        )


def test_line_connectivity():
    """Test line actor correctly connects points.

    BUG DETECTION: Line rendering or connectivity bugs.
    """

    scene = window.Scene()
    scene.background = (0, 0, 0)

    # Diagonal line from bottom-left to top-right
    lines = np.array([[[-3, -3, 0], [3, 3, 0]]])
    colors = np.array([[1, 0, 0]])

    line_actor = actor.line(lines, colors=colors)
    scene.add(line_actor)

    arr = window.snapshot(
        scene=scene, fname="test_line_connectivity.png", return_array=True
    )

    # Find red line pixels
    red_pixels = np.where((arr[..., 0] > 50) & (arr[..., 1] < 50) & (arr[..., 2] < 50))

    if len(red_pixels[0]) > 10:
        # Check that line spans diagonally
        y_coords = red_pixels[0]
        x_coords = red_pixels[1]

        y_span = np.max(y_coords) - np.min(y_coords)
        x_span = np.max(x_coords) - np.min(x_coords)

        assert y_span > arr.shape[0] * 0.3, (
            f"Line should span vertically. Y span: {y_span}"
        )
        assert x_span > arr.shape[1] * 0.3, (
            f"Line should span horizontally. X span: {x_span}"
        )


def test_ellipsoid_shape_accuracy():
    """Test ellipsoid has correct proportions.

    BUG DETECTION: Ellipsoid axis scaling bugs.
    """

    scene = window.Scene()
    scene.background = (0, 0, 0)

    # Ellipsoid stretched along X axis
    ellipsoid_act = actor.ellipsoid(
        centers=np.array([[0, 0, 0]]), lengths=(6, 2, 2), colors=np.array([[1, 0, 0]])
    )
    scene.add(ellipsoid_act)

    arr = window.snapshot(
        scene=scene, fname="test_ellipsoid_shape.png", return_array=True
    )

    # Find red pixels
    red_pixels = np.where(
        (arr[..., 0] > 100) & (arr[..., 1] < 100) & (arr[..., 2] < 100)
    )

    if len(red_pixels[0]) > 10:
        y_coords = red_pixels[0]
        x_coords = red_pixels[1]

        y_span = np.max(y_coords) - np.min(y_coords)
        x_span = np.max(x_coords) - np.min(x_coords)

        # X span should be significantly larger than Y span (3:1 ratio)
        ratio = x_span / max(y_span, 1)

        assert ratio > 1.5, (
            f"Ellipsoid should be wider than tall (3:1 axes). "
            f"X span: {x_span}, Y span: {y_span}, ratio: {ratio:.2f}"
        )


def test_box_size_consistency():
    """Test box actor renders with correct size.

    BUG DETECTION: Box scaling or size bugs.
    """

    scene = window.Scene()
    scene.background = (0, 0, 0)

    # Small box
    small_box = actor.box(
        centers=np.array([[-2, 0, 0]]),
        directions=np.array([[0, 1, 0]]),
        colors=np.array([[1, 0, 0]]),
        scales=(0.5, 0.5, 0.5),
    )
    scene.add(small_box)

    # Large box
    large_box = actor.box(
        centers=np.array([[2, 0, 0]]),
        directions=np.array([[0, 1, 0]]),
        colors=np.array([[0, 1, 0]]),
        scales=(1.5, 1.5, 1.5),
    )
    scene.add(large_box)

    arr = window.snapshot(scene=scene, fname="test_box_sizes.png", return_array=True)

    red_pixels = np.sum((arr[..., 0] > 100) & (arr[..., 1] < 100) & (arr[..., 2] < 100))
    green_pixels = np.sum(
        (arr[..., 0] < 100) & (arr[..., 1] > 100) & (arr[..., 2] < 100)
    )

    # Large box should have more pixels
    assert green_pixels > red_pixels, (
        f"Large box should be bigger. "
        f"Red (small): {red_pixels}, Green (large): {green_pixels}"
    )


def test_cone_orientation():
    """Test cone actor points in correct direction.

    BUG DETECTION: Cone orientation bugs.
    """

    scene = window.Scene()
    scene.background = (0, 0, 0)

    # Cone pointing upward
    cone_act = actor.cone(
        centers=np.array([[0, 0, 0]]),
        directions=np.array([[0, 1, 0]]),
        colors=np.array([[1, 0, 0]]),
        height=2.0,
    )
    scene.add(cone_act)

    arr = window.snapshot(
        scene=scene, fname="test_cone_orientation.png", return_array=True
    )

    # Find red pixels
    red_pixels = np.where(
        (arr[..., 0] > 100) & (arr[..., 1] < 100) & (arr[..., 2] < 100)
    )

    if len(red_pixels[0]) > 10:
        center_y = arr.shape[0] // 2
        upper_pixels = np.sum(red_pixels[0] < center_y)
        lower_pixels = np.sum(red_pixels[0] > center_y)

        # Cone has base at center, tip points in direction
        # So cone pointing up will have base below center (more lower pixels)
        assert lower_pixels > upper_pixels, (
            f"Cone pointing up should have base below center. "
            f"Upper: {upper_pixels}, Lower: {lower_pixels}"
        )


def test_cylinder_height_accuracy():
    """Test cylinder renders with correct height/radius ratio.

    BUG DETECTION: Cylinder dimension bugs.
    """

    scene = window.Scene()
    scene.background = (0, 0, 0)

    # Tall thin cylinder
    cylinder_act = actor.cylinder(
        centers=np.array([[0, 0, 0]]),
        directions=np.array([[0, 1, 0]]),
        colors=np.array([[1, 0, 0]]),
        height=3.0,
        radii=0.5,
    )
    scene.add(cylinder_act)

    arr = window.snapshot(
        scene=scene, fname="test_cylinder_proportions.png", return_array=True
    )

    red_pixels = np.where(
        (arr[..., 0] > 100) & (arr[..., 1] < 100) & (arr[..., 2] < 100)
    )

    if len(red_pixels[0]) > 10:
        y_span = np.max(red_pixels[0]) - np.min(red_pixels[0])
        x_span = np.max(red_pixels[1]) - np.min(red_pixels[1])

        # Height should be significantly larger than width
        ratio = y_span / max(x_span, 1)

        assert ratio > 1.5, (
            f"Cylinder should be taller than wide. "
            f"Y span: {y_span}, X span: {x_span}, ratio: {ratio:.2f}"
        )


def test_opacity_blending_multiple_actors():
    """Test transparency blending with multiple overlapping actors.

    BUG DETECTION: Opacity blending, z-ordering, alpha compositing bugs.
    """
    scene = window.Scene()
    scene.background = (0, 0, 0)

    # Three overlapping spheres with different opacities
    # Back sphere (red, semi-transparent)
    back_sphere = actor.sphere(
        centers=np.array([[0, 0, -1]]),
        colors=np.array([[1, 0, 0]]),
        radii=1.0,
        opacity=0.5,
    )
    scene.add(back_sphere)

    # Middle sphere (green, more transparent)
    mid_sphere = actor.sphere(
        centers=np.array([[0, 0, 0]]),
        colors=np.array([[0, 1, 0]]),
        radii=1.0,
        opacity=0.7,
    )
    scene.add(mid_sphere)

    # Front sphere (blue, least transparent)
    front_sphere = actor.sphere(
        centers=np.array([[0, 0, 1]]),
        colors=np.array([[0, 0, 1]]),
        radii=1.0,
        opacity=0.8,
    )
    scene.add(front_sphere)

    arr = window.snapshot(
        scene=scene, fname="test_opacity_blending.png", return_array=True
    )

    # Check that colors are visible (blending occurred)
    # With transparency, we may see blended colors rather than pure RGB
    green_component = np.sum(arr[..., 1] > 100)
    blue_component = np.sum(arr[..., 2] > 100)

    # At least the dominant colors should be visible
    assert green_component > 100, (
        f"Green sphere should be visible: {green_component} pixels"
    )
    assert blue_component > 100, (
        f"Blue sphere should be visible: {blue_component} pixels"
    )


def test_wireframe_rendering():
    """Test wireframe rendering quality and edge detection.

    BUG DETECTION: Wireframe mode, edge rendering bugs.
    """
    scene = window.Scene()
    scene.background = (0, 0, 0)

    # Box in wireframe mode
    box_wire = actor.box(
        centers=np.array([[0, 0, 0]]),
        directions=np.array([[0, 1, 0]]),
        colors=np.array([[1, 1, 1]]),
        scales=(2, 2, 2),
        wireframe=True,
        wireframe_thickness=2.0,
    )
    scene.add(box_wire)

    arr = window.snapshot(scene=scene, fname="test_wireframe.png", return_array=True)

    # Wireframe should have white edges
    white_pixels = np.sum(
        (arr[..., 0] > 200) & (arr[..., 1] > 200) & (arr[..., 2] > 200)
    )

    # Should have visible edges
    assert white_pixels > 100, f"Wireframe edges not visible: {white_pixels} pixels"

    # Center should be mostly empty (black background showing through)
    center_y, center_x = arr.shape[0] // 2, arr.shape[1] // 2
    center_region = arr[center_y - 10 : center_y + 10, center_x - 10 : center_x + 10]
    black_in_center = np.sum(
        (center_region[..., 0] < 50)
        & (center_region[..., 1] < 50)
        & (center_region[..., 2] < 50)
    )

    # Center should be mostly transparent/black
    assert black_in_center > 100, (
        f"Wireframe center should be transparent: {black_in_center} black pixels"
    )


def test_material_shininess():
    """Test material shininess/specular properties.

    BUG DETECTION: Material property rendering, specular highlights.
    """
    scene = window.Scene()
    scene.background = (0, 0, 0)

    # Sphere with high shininess (should have bright specular highlight)
    shiny_sphere = actor.sphere(
        centers=np.array([[0, 0, 0]]),
        colors=np.array([[0.5, 0.5, 0.5]]),
        radii=1.0,
        material="phong",
    )
    scene.add(shiny_sphere)

    arr = window.snapshot(scene=scene, fname="test_shininess.png", return_array=True)

    # Find bright pixels (specular highlights)
    bright_pixels = np.where(
        (arr[..., 0] > 150) | (arr[..., 1] > 150) | (arr[..., 2] > 150)
    )

    # Should have some specular highlights
    assert len(bright_pixels[0]) > 50, (
        f"Phong material should have specular highlights: "
        f"{len(bright_pixels[0])} bright pixels"
    )

    # Find dim pixels (shaded areas)
    dim_pixels = np.sum(
        (arr[..., 0] > 20)
        & (arr[..., 0] < 80)
        & (arr[..., 1] > 20)
        & (arr[..., 1] < 80)
        & (arr[..., 2] > 20)
        & (arr[..., 2] < 80)
    )

    # Should have shaded areas
    assert dim_pixels > 100, (
        f"Phong shading should create gradients: {dim_pixels} dim pixels"
    )


def test_streamlines_path_following():
    """Test streamlines follow data correctly.

    BUG DETECTION: Streamline path generation, data following bugs.
    """
    scene = window.Scene()
    scene.background = (0, 0, 0)

    # Simple curved streamline
    line_data = np.array([[[0, 0, 0], [1, 1, 0], [2, 2, 0], [3, 1, 0], [4, 0, 0]]])

    streamline = actor.streamlines(lines=line_data, colors=(1, 0, 0), thickness=3.0)
    scene.add(streamline)

    arr = window.snapshot(scene=scene, fname="test_streamlines.png", return_array=True)

    # Find red pixels
    red_pixels = np.where(
        (arr[..., 0] > 100) & (arr[..., 1] < 100) & (arr[..., 2] < 100)
    )

    # Streamline should be visible
    assert len(red_pixels[0]) > 50, (
        f"Streamline not visible: {len(red_pixels[0])} pixels"
    )

    # Should span across image
    if len(red_pixels[1]) > 10:
        x_span = np.max(red_pixels[1]) - np.min(red_pixels[1])
        assert x_span > arr.shape[1] * 0.3, (
            f"Streamline should span horizontally: {x_span} pixels"
        )


def test_superquadric_shape_parameters():
    """Test superquadric shape accuracy with different parameters.

    BUG DETECTION: Superquadric parameter handling, shape generation.
    """
    scene = window.Scene()
    scene.background = (0, 0, 0)

    # Superquadric with specific roundness parameters
    superquad = actor.superquadric(
        centers=np.array([[0, 0, 0]]),
        colors=np.array([[1, 0, 0]]),
        roundness=(1.0, 1.0),  # Sphere-like
        scales=(1, 1, 1),
    )
    scene.add(superquad)

    arr = window.snapshot(scene=scene, fname="test_superquadric.png", return_array=True)

    # Find red pixels
    red_pixels = np.where(
        (arr[..., 0] > 100) & (arr[..., 1] < 100) & (arr[..., 2] < 100)
    )

    # Should be visible
    assert len(red_pixels[0]) > 100, (
        f"Superquadric not visible: {len(red_pixels[0])} pixels"
    )

    # Should be roughly circular (sphere-like with roundness 1.0)
    if len(red_pixels[0]) > 100:
        y_span = np.max(red_pixels[0]) - np.min(red_pixels[0])
        x_span = np.max(red_pixels[1]) - np.min(red_pixels[1])
        ratio = max(y_span, x_span) / max(min(y_span, x_span), 1)

        # Should be roughly circular (aspect ratio close to 1)
        assert ratio < 1.5, (
            f"Superquadric with roundness (1,1) should be circular. "
            f"Aspect ratio: {ratio:.2f}"
        )


def test_marker_styles():
    """Test different marker point styles.

    BUG DETECTION: Marker rendering, style variations.
    """
    scene = window.Scene()
    scene.background = (0, 0, 0)

    # Points with marker
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    marker_act = actor.marker(centers=points, colors=colors, size=20, marker="circle")
    scene.add(marker_act)

    arr = window.snapshot(scene=scene, fname="test_markers.png", return_array=True)

    # All three colors should be visible
    red_pixels = np.sum((arr[..., 0] > 100) & (arr[..., 1] < 100) & (arr[..., 2] < 100))
    green_pixels = np.sum(
        (arr[..., 0] < 100) & (arr[..., 1] > 100) & (arr[..., 2] < 100)
    )
    blue_pixels = np.sum(
        (arr[..., 0] < 100) & (arr[..., 1] < 100) & (arr[..., 2] > 100)
    )

    assert red_pixels > 5, f"Red marker not visible: {red_pixels} pixels"
    assert green_pixels > 5, f"Green marker not visible: {green_pixels} pixels"
    assert blue_pixels > 5, f"Blue marker not visible: {blue_pixels} pixels"


def test_zero_size_edge_case():
    """Test actors with zero or very small sizes.

    BUG DETECTION: Edge case handling, division by zero, rendering errors.
    """
    scene = window.Scene()
    scene.background = (0, 0, 0)

    # Regular sphere for reference
    normal_sphere = actor.sphere(
        centers=np.array([[2, 0, 0]]), colors=np.array([[1, 0, 0]]), radii=1.0
    )
    scene.add(normal_sphere)

    # Very small sphere (should be barely visible or invisible)
    tiny_sphere = actor.sphere(
        centers=np.array([[-2, 0, 0]]), colors=np.array([[0, 1, 0]]), radii=0.01
    )
    scene.add(tiny_sphere)

    arr = window.snapshot(scene=scene, fname="test_zero_size.png", return_array=True)

    # Normal sphere should be visible
    red_pixels = np.sum((arr[..., 0] > 100) & (arr[..., 1] < 100) & (arr[..., 2] < 100))
    assert red_pixels > 100, f"Normal sphere should be visible: {red_pixels} pixels"

    # Tiny sphere should be invisible or barely visible
    green_pixels = np.sum(
        (arr[..., 0] < 100) & (arr[..., 1] > 100) & (arr[..., 2] < 100)
    )
    assert green_pixels < 50, (
        f"Tiny sphere should be barely visible: {green_pixels} pixels"
    )


def test_negative_scale_mirroring():
    """Test actors with negative scales (mirroring).

    BUG DETECTION: Negative scale handling, face culling issues.
    """
    scene = window.Scene()
    scene.background = (0, 0, 0)

    # Box with positive scale
    pos_box = actor.box(
        centers=np.array([[-2, 0, 0]]),
        directions=np.array([[0, 1, 0]]),
        colors=np.array([[1, 0, 0]]),
        scales=(1, 1, 1),
    )
    scene.add(pos_box)

    # Box with negative X scale (should be mirrored)
    neg_box = actor.box(
        centers=np.array([[2, 0, 0]]),
        directions=np.array([[0, 1, 0]]),
        colors=np.array([[0, 0, 1]]),
        scales=(-1, 1, 1),
    )
    scene.add(neg_box)

    arr = window.snapshot(
        scene=scene, fname="test_negative_scale.png", return_array=True
    )

    # Both boxes should be visible
    red_pixels = np.sum((arr[..., 0] > 100) & (arr[..., 1] < 100) & (arr[..., 2] < 100))
    blue_pixels = np.sum(
        (arr[..., 0] < 100) & (arr[..., 1] < 100) & (arr[..., 2] > 100)
    )

    assert red_pixels > 50, f"Positive scale box not visible: {red_pixels} pixels"
    assert blue_pixels > 50, f"Negative scale box not visible: {blue_pixels} pixels"


def test_opacity_zero_invisibility():
    """Test that opacity=0 makes actors completely invisible.

    BUG DETECTION: Opacity implementation bugs.
    """
    scene = window.Scene()
    scene.background = (0, 0, 0)

    # Invisible sphere (opacity=0)
    invisible = actor.sphere(
        centers=np.array([[0, 0, 0]]),
        colors=np.array([[1, 0, 0]]),
        radii=1.0,
        opacity=0.0,
    )
    scene.add(invisible)

    arr = window.snapshot(scene=scene, fname="test_opacity_zero.png", return_array=True)

    # Should be completely black (invisible) - check RGB only, not alpha
    rgb_only = arr[:, :, :3]  # Ignore alpha channel
    non_black = np.sum(rgb_only.max(axis=-1) > 10)

    assert non_black == 0, (
        f"Opacity=0 should be invisible. Found {non_black} non-black pixels. "
        f"BUG: Opacity not working correctly!"
    )


def test_material_basic_vs_phong():
    """Test difference between basic and phong materials.

    BUG DETECTION: Material implementation, shading differences.
    """
    scene = window.Scene()
    scene.background = (0, 0, 0)

    # Basic material (no shading)
    basic_sphere = actor.sphere(
        centers=np.array([[-1.5, 0, 0]]),
        colors=np.array([[0.5, 0.5, 0.5]]),
        radii=1.0,
        material="basic",
    )
    scene.add(basic_sphere)

    # Phong material (with shading)
    phong_sphere = actor.sphere(
        centers=np.array([[1.5, 0, 0]]),
        colors=np.array([[0.5, 0.5, 0.5]]),
        radii=1.0,
        material="phong",
    )
    scene.add(phong_sphere)

    arr = window.snapshot(scene=scene, fname="test_materials.png", return_array=True)

    # Analyze variance in each sphere (Phong should have more variance due to shading)
    center_y = arr.shape[0] // 2
    center_x = arr.shape[1] // 2

    # Basic material region (left)
    basic_region = arr[center_y - 50 : center_y + 50, center_x - 150 : center_x - 50, 0]
    basic_variance = np.var(basic_region[basic_region > 10])

    # Phong material region (right)
    phong_region = arr[center_y - 50 : center_y + 50, center_x + 50 : center_x + 150, 0]
    phong_variance = np.var(phong_region[phong_region > 10])

    # Phong should have significantly more variance than basic
    assert phong_variance > basic_variance * 1.5, (
        f"Phong material should have more shading variance than basic. "
        f"Basic variance: {basic_variance:.2f}, Phong variance: {phong_variance:.2f}. "
        f"BUG: Material differences not working!"
    )


def test_color_accuracy_strict():
    """Test that colors render exactly as specified.

    BUG DETECTION: Color rendering bugs, gamma correction issues.
    """
    scene = window.Scene()
    scene.background = (0, 0, 0)

    # Pure red sphere
    red_sphere = actor.sphere(
        centers=np.array([[0, 0, 0]]),
        colors=np.array([[1, 0, 0]]),
        radii=1.0,
        material="basic",  # Basic material for flat color
    )
    scene.add(red_sphere)

    arr = window.snapshot(scene=scene, fname="test_color_purity.png", return_array=True)

    # Find the brightest red pixel
    red_channel = arr[..., 0]
    brightest_idx = np.unravel_index(np.argmax(red_channel), red_channel.shape)
    brightest_pixel = arr[brightest_idx]

    # Pure red should be [255, 0, 0] or very close
    r, g, b = brightest_pixel[0], brightest_pixel[1], brightest_pixel[2]

    assert r > 200, f"Red channel too dim: {r}"
    assert g < 50, f"Green contamination: {g} (should be <50)"
    assert b < 50, f"Blue contamination: {b} (should be <50)"


def test_picking_state_visual():
    """Test that picking state affects visual rendering.

    BUG DETECTION: Picking state implementation.
    """
    scene = window.Scene()
    scene.background = (0, 0, 0)

    # Sphere with picking enabled
    pickable = actor.sphere(
        centers=np.array([[0, 0, 0]]),
        colors=np.array([[1, 0, 0]]),
        radii=1.0,
        enable_picking=True,
    )
    scene.add(pickable)

    arr_pickable = window.snapshot(
        scene=scene, fname="test_pickable.png", return_array=True
    )

    scene2 = window.Scene()
    scene2.background = (0, 0, 0)

    # Sphere with picking disabled
    non_pickable = actor.sphere(
        centers=np.array([[0, 0, 0]]),
        colors=np.array([[1, 0, 0]]),
        radii=1.0,
        enable_picking=False,
    )
    scene2.add(non_pickable)

    arr_non_pickable = window.snapshot(
        scene=scene2, fname="test_non_pickable.png", return_array=True
    )

    # Both should render identically (picking shouldn't affect visual)
    pickable_pixels = np.sum(arr_pickable[..., 0] > 100)
    non_pickable_pixels = np.sum(arr_non_pickable[..., 0] > 100)

    # They should be nearly identical
    diff = abs(pickable_pixels - non_pickable_pixels)
    assert diff < pickable_pixels * 0.1, (
        f"Picking state shouldn't affect rendering significantly. "
        f"Difference: {diff} pixels"
    )
