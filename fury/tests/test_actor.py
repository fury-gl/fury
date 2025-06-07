import re

from PIL import Image
import numpy as np
import numpy.testing as npt
import pytest

from fury import actor, window
from fury.lib import Group
from fury.material import (
    VectorFieldArrowMaterial,
    VectorFieldLineMaterial,
    VectorFieldThinLineMaterial,
)
from fury.utils import get_slices, set_group_visibility, show_slices


def validate_actors(actor_type="actor_name", prim_count=1, **kwargs):
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


def test_sphere():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="sphere")


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


def test_box():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="box")


def test_cylinder():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="cylinder")


def test_square():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="square")


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


def test_superquadric():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="superquadric")


def test_cone():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="cone")


def test_star():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="star")


def test_disk():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="disk")
    validate_actors(centers=centers, colors=colors, actor_type="disk", sectors=8)


def test_triangle():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="triangle")


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


def test_valid_3d_data():
    """Test valid 3D input with default parameters (Test Case 1)."""
    data = np.random.rand(10, 20, 30)
    slicer_obj = actor.slicer(data)

    # Verify object type and visibility
    assert isinstance(slicer_obj, Group)
    assert slicer_obj.visible
    assert len(slicer_obj.children) == 3
    assert all(child.visible for child in slicer_obj.children)


def test_invalid_4d_data():
    """Test invalid 4D data shape (Test Case 4)."""
    data = np.random.rand(10, 20, 30, 4)  # Last dim ≠ 3
    with pytest.raises(ValueError) as excinfo:
        actor.slicer(data)
    assert "Last dimension must be of size 3" in str(excinfo.value)


def test_opacity_validation():
    """Test opacity validation raises errors for out-of-bounds values"""
    data = np.random.rand(10, 20, 30)

    # Test valid values
    for valid_opacity in [0, 0.5, 1]:
        slicer_obj = actor.slicer(data, opacity=valid_opacity)
        for child in slicer_obj.children:
            assert child.material.opacity == valid_opacity

    # Test invalid values
    for invalid_opacity in [-0.1, 1.1, 2.0]:
        with pytest.raises(ValueError) as excinfo:
            actor.slicer(data, opacity=invalid_opacity)
        assert "Opacity must be between 0 and 1" in str(excinfo.value)


def test_custom_initial_slices():
    """Test custom initial slice positions (Test Case 10)."""
    data = np.random.rand(10, 20, 30)
    slicer_obj = actor.slicer(data, initial_slices=(5, 10, 15))

    # Verify slice positions match input
    assert np.array_equal(get_slices(slicer_obj), [5, 10, 15])

    # Verify positions update correctly
    show_slices(slicer_obj, (2, 4, 6))
    assert np.array_equal(get_slices(slicer_obj), [2, 4, 6])


def test_visibility_control():
    """Test visibility settings through methods (Test Case 13)."""
    data = np.random.rand(10, 20, 30)
    slicer_obj = actor.slicer(data, visibility=(True, True, True))

    # Verify initial visibility
    assert all(child.visible for child in slicer_obj.children)

    # Update and verify new visibility
    set_group_visibility(slicer_obj, (False, True, False))
    visibilities = [child.visible for child in slicer_obj.children]
    assert visibilities == [False, True, False]


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
    assert np.all(vf.cross_section == np.array([-2, -2, -2]))

    # Test setting cross section
    # cross section will not work without providing visibility.
    new_cross = [1, 2, 3]
    vf.cross_section = new_cross
    assert np.all(vf.cross_section == np.array([-2, -2, -2]))

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
    assert vf.visibility == (True, False, True)

    # Set cross section with visibility
    vf.cross_section = [1, 2, 3]
    # The y dimension should be -1 because visibility[1] is False
    assert np.all(vf.cross_section == np.array([1, -1, 3]))


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
    assert np.all(vf.cross_section == np.array([2, -1, 2]))


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
    assert vf.material.thickness == 0.0
