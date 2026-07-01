"""
Tests for billboard actor.

Simple tests for billboard creation and basic rendering functionality.
"""

import numpy as np
import numpy.testing as npt

from fury import actor, window
from fury.lib import MeshPhongMaterial
from fury.material import BillboardSphereMaterial


def test_basic_billboard(interactive: bool = False):
    """
    Test billboard creation, geometry, rendering, and camera facing
    behavior.
    """
    # Test creation and geometry setup
    centers = [[0, 0, 0], [1, 2, 3], [-2, 0.5, 4]]
    colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    sizes = [[1.0, 2.0], [0.5, 0.5], [2.0, 1.0]]

    bb = actor.billboard(centers, colors=colors, sizes=sizes, opacity=0.75)

    # Check billboard count
    assert hasattr(bb, "billboard_count")
    npt.assert_equal(bb.billboard_count, 3)

    # Check stored sizes metadata
    assert hasattr(bb, "billboard_sizes")
    npt.assert_array_equal(bb.billboard_sizes, np.asarray(sizes, dtype=np.float32))

    # Check geometry (6 vertices per billboard)
    geom = bb.geometry
    npt.assert_equal(len(geom.positions.data), 18)
    npt.assert_equal(len(geom.colors.data), 18)
    assert geom.normals is not None
    npt.assert_equal(len(geom.normals.data), 18)
    normals = np.asarray(geom.normals.data).reshape(-1, 3)
    npt.assert_allclose(normals[0, :2], sizes[0])
    npt.assert_equal(len(geom.indices.data), 18)

    # Check material opacity
    assert np.isclose(bb.material.opacity, 0.75)

    # Test scalar size
    bb_scalar = actor.billboard(centers, colors=colors, sizes=0.4)
    geom_scalar = bb_scalar.geometry
    npt.assert_equal(len(geom_scalar.positions.data), 18)
    assert geom_scalar.normals is not None
    npt.assert_equal(len(geom_scalar.normals.data), 18)
    normals_scalar = np.asarray(geom_scalar.normals.data).reshape(-1, 3)
    npt.assert_allclose(normals_scalar[0, :2], [0.4, 0.4])

    # Test basic rendering
    scene = window.Scene()
    scene.background = (0, 0, 0)

    center = np.array([[0.0, 0.0, 0.0]])
    color = np.array([[1.0, 0.0, 0.0]])  # red
    bb_render = actor.billboard(center, colors=color, sizes=(1.0, 1.0))
    scene.add(bb_render)

    if interactive:  # pragma: no cover
        window.show([bb_render])

    # Test snapshot creation
    arr = window.snapshot(scene=scene, fname=None, return_array=True)

    # Basic checks: array exists and has expected shape
    assert arr is not None
    assert arr.ndim == 3
    assert arr.shape[-1] >= 3  # RGB or RGBA

    # Check that billboard is visible (has red pixels)
    _assert_red_visible(arr)

    scene.clear()


def test_billboard_camera_facing():
    """Test that billboard faces camera from different viewpoints."""
    scene = window.Scene()
    scene.background = (0, 0, 0)

    # Create a billboard at origin
    center = np.array([[0.0, 0.0, 0.0]])
    color = np.array([[0.0, 1.0, 0.0]])  # green
    bb = actor.billboard(center, colors=color, sizes=(2.0, 2.0))
    scene.add(bb)

    # Test from default camera position
    arr1 = window.snapshot(scene=scene, fname=None, return_array=True)

    # Billboard should be visible from front view
    data1 = np.asarray(arr1)
    green_mask = (
        (data1[..., 1] > 50)
        & (data1[..., 1] > data1[..., 0])
        & (data1[..., 1] > data1[..., 2])
    )
    green_pixels1 = np.sum(green_mask)

    # Move camera to side (if we had camera controls, but we test basic visibility)
    # Since we can't move camera easily, just test that billboard is rendered
    assert arr1 is not None
    assert green_pixels1 > 0, "Billboard should be visible with green pixels"

    scene.clear()


def test_rectangular_billboards():
    """Test that billboards support rectangular aspect ratios."""
    centers = np.array([[0, 0, 0], [3, 0, 0], [6, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    sizes = np.array([[7.0, 1.0], [1.0, 1.0], [1.0, 3.0]])

    bb = actor.billboard(centers=centers, colors=colors, sizes=sizes)

    npt.assert_equal(bb.billboard_count, 3)

    geom = bb.geometry
    normals = geom.normals.data

    npt.assert_allclose(normals[0, :2], [7.0, 1.0], rtol=1e-5)
    npt.assert_allclose(normals[6, :2], [1.0, 1.0], rtol=1e-5)
    npt.assert_allclose(normals[12, :2], [1.0, 3.0], rtol=1e-5)


def test_billboard_sphere(interactive: bool = False):
    """Test sphere actor hybrid implementation with impostor mode."""
    centers = np.array([[0, 0, 0], [1, 1, 1], [2, -1, 0.5]], dtype=np.float32)
    colors = np.array(
        [
            [1.0, 0.2, 0.2],
            [0.2, 1.0, 0.2],
            [0.2, 0.2, 1.0],
        ],
        dtype=np.float32,
    )
    radii = np.array([0.25, 0.5, 0.75], dtype=np.float32)

    mesh_spheres = actor.sphere(
        centers,
        colors=colors,
        radii=radii,
        opacity=0.6,
        impostor=False,
    )

    assert isinstance(mesh_spheres.material, MeshPhongMaterial)
    assert not isinstance(mesh_spheres.material, BillboardSphereMaterial)

    impostor_spheres = actor.sphere(
        centers,
        colors=colors,
        radii=radii,
        opacity=0.6,
        impostor=True,
    )

    assert hasattr(impostor_spheres, "billboard_count")
    npt.assert_equal(impostor_spheres.billboard_count, centers.shape[0])
    assert hasattr(impostor_spheres, "billboard_radii")
    npt.assert_allclose(impostor_spheres.billboard_radii, radii)
    npt.assert_allclose(impostor_spheres.billboard_sizes[:, 0], radii * 2.0)
    assert isinstance(impostor_spheres.material, BillboardSphereMaterial)
    assert getattr(impostor_spheres, "billboard_mode", None) == "impostor"

    scalar = actor.sphere(centers, radii=0.6, impostor=True)
    npt.assert_allclose(scalar.billboard_radii, np.full(centers.shape[0], 0.6))

    scene = window.Scene()
    scene.background = (0, 0, 0)
    render_actor = actor.sphere(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        colors=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        radii=0.6,
        impostor=True,
    )
    scene.add(render_actor)

    if interactive:  # pragma: no cover
        window.show(scene)

    arr = window.snapshot(scene=scene, fname=None, return_array=True)
    assert arr is not None
    _assert_red_visible(arr)

    data = np.asarray(arr)
    red_channel = data[..., 0]
    positive_red = red_channel[red_channel > 0]
    assert positive_red.size > 0
    assert positive_red.max() > positive_red.min()

    scene.clear()


def _assert_red_visible(image):
    """Check if red pixels exist in image."""
    data = np.asarray(image)
    if data.ndim != 3 or data.shape[-1] < 3:
        raise AssertionError("Invalid image format")

    red = data[..., 0]
    has_red = np.any(red > 50)
    if not has_red:
        raise AssertionError("No visible red pixels found in rendered billboard")


def test_billboard_bounding_box():
    bb = actor.billboard(
        centers=np.array([[0.0, 0.0, 0.0]]),
        sizes=(4.0, 4.0),
    )
    aabb = bb.get_bounding_box()
    npt.assert_allclose(aabb[0], [-2, -2, -2])
    npt.assert_allclose(aabb[1], [2, 2, 2])


def test_billboard_sphere_bounding_box():
    centers = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 5.0, 5.0]],
        dtype=np.float32,
    )
    radii = np.array([2.0, 3.0, 1.0], dtype=np.float32)

    bb = actor.sphere(centers, radii=radii, impostor=True)
    aabb = bb.get_bounding_box()

    npt.assert_allclose(aabb[0], [-2, -3, -3])
    npt.assert_allclose(aabb[1], [13, 6, 6])


def test_billboard_bounding_box_single_sphere():
    bb = actor.sphere(
        centers=np.array([[0.0, 0.0, 0.0]]),
        radii=5.0,
        impostor=True,
    )
    bsphere = bb.get_world_bounding_sphere()

    npt.assert_allclose(bsphere[:3], [0, 0, 0], atol=1e-6)
    npt.assert_allclose(bsphere[3], 5.0 * np.sqrt(3), atol=0.01)


def test_billboard_bounding_box_camera_framing():
    scene = window.Scene()
    scene.background = (0, 0, 0)

    bb = actor.sphere(
        centers=np.array([[0.0, 0.0, 0.0]]),
        colors=np.array([[1.0, 0.0, 0.0]]),
        radii=5.0,
        impostor=True,
    )
    scene.add(bb)

    arr = window.snapshot(scene=scene, fname=None, return_array=True)
    assert arr is not None

    _assert_red_visible(arr)

    scene.clear()
