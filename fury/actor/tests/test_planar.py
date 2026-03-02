from PIL import Image
import numpy as np
import numpy.testing as npt
import pytest

from fury import actor, window
from fury.actor.tests._helpers import validate_actors


def test_square():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="square")


def test_disk():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="disk")
    validate_actors(centers=centers, colors=colors, actor_type="disk", sectors=8)


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


def test_image():
    scene = window.Scene()
    image = np.random.rand(100, 100)
    position = np.array([10, 10, 10])
    image_actor = actor.image(image=image, position=position)
    scene.add(image_actor)

    npt.assert_array_equal(image_actor.local.position, position)
    assert image_actor.visible

    scene.remove(image_actor)


def test_star():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="star")


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
