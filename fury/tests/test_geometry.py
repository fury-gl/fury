import numpy as np
import numpy.testing as npt
import pytest

from fury import geometry


def test_buffer_to_geometry():
    positions = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]).astype("float32")
    geo = geometry.buffer_to_geometry(positions)
    npt.assert_array_equal(geo.positions.view, positions)

    normals = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]).astype("float32")
    colors = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]).astype("float32")
    indices = np.array([[0, 1, 2]]).astype("int32")
    geo = geometry.buffer_to_geometry(
        positions, colors=colors, normals=normals, indices=indices
    )

    npt.assert_array_equal(geo.colors.view, colors)
    npt.assert_array_equal(geo.normals.view, normals)
    npt.assert_array_equal(geo.indices.view, indices)


def test_line_buffer_separator():
    line_vertices = [
        np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
        np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=np.float32),
    ]
    positions, colors = geometry.line_buffer_separator(line_vertices)
    # Check positions
    npt.assert_array_equal(positions[:2], line_vertices[0])
    assert np.all(np.isnan(positions[2]))
    npt.assert_array_equal(positions[3:], line_vertices[1])
    expected_colors = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [np.nan, np.nan, np.nan, np.nan],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    npt.assert_array_equal(colors, expected_colors)

    line_vertices = [
        np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
        np.array([[2, 2, 2], [3, 3, 3]], dtype=np.float32),
    ]
    color = np.array([[1.0, 0, 0], [0.0, 1.0, 0.0]], dtype=np.float32)
    positions, colors = geometry.line_buffer_separator(line_vertices, color=color)
    expected_colors = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [np.nan, np.nan, np.nan],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )
    npt.assert_array_equal(colors, expected_colors)

    line_vertices = [
        np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
        np.array([[2, 2, 2], [3, 3, 3]], dtype=np.float32),
    ]
    color = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float32)
    positions, colors = geometry.line_buffer_separator(line_vertices, color=color)
    expected_colors = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [np.nan, np.nan, np.nan],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.float32,
    )
    npt.assert_array_equal(colors, expected_colors)

    line_vertices = [
        np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
        np.array([[2, 2, 2], [3, 3, 3]], dtype=np.float32),
    ]
    color = np.array([[1.0, 0, 0], [0.0, 1.0, 0.0]], dtype=np.float32)
    positions, colors = geometry.line_buffer_separator(line_vertices, color=color)
    expected_colors = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [np.nan, np.nan, np.nan],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )
    npt.assert_array_equal(colors, expected_colors)

    line_vertices = [
        np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
        np.array([[2, 2, 2], [3, 3, 3]], dtype=np.float32),
    ]
    color = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float32)
    positions, colors = geometry.line_buffer_separator(line_vertices, color=color)
    expected_colors = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [np.nan, np.nan, np.nan],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.float32,
    )
    npt.assert_array_equal(colors, expected_colors)

    line_vertices = [
        np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
        np.array([[2, 2, 2], [3, 3, 3]], dtype=np.float32),
    ]
    color = [
        np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        np.array([[0, 0, 1], [1, 1, 0]], dtype=np.float32),
    ]
    positions, colors = geometry.line_buffer_separator(line_vertices, color=color)
    expected_colors = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [np.nan, np.nan, np.nan],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.float32,
    )
    npt.assert_array_equal(colors, expected_colors)

    line_vertices = [
        np.array([[0, 0, 0]], dtype=np.float32),
        np.array([[1, 1, 1]], dtype=np.float32),
    ]
    color = np.array([1, 0, 0], dtype=np.float32)  # valid color now

    geometry.line_buffer_separator(line_vertices, color=color)

    line_vertices = [np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)]
    positions, colors = geometry.line_buffer_separator(line_vertices)
    npt.assert_array_equal(positions, line_vertices[0])
    expected_colors = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    npt.assert_array_equal(colors, expected_colors)

    line_vertices = [np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)]
    color = np.array([[1, 0, 0]], dtype=np.float32)
    positions, colors = geometry.line_buffer_separator(line_vertices, color=color)
    expected_colors = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)
    npt.assert_array_equal(colors, expected_colors)


def _is_orthonormal(d, x, y, atol=1e-6):
    return (
        np.isclose(np.linalg.norm(d), 1.0, atol=atol)
        and np.isclose(np.linalg.norm(x), 1.0, atol=atol)
        and np.isclose(np.linalg.norm(y), 1.0, atol=atol)
        and np.isclose(np.dot(d, x), 0.0, atol=atol)
        and np.isclose(np.dot(d, y), 0.0, atol=atol)
        and np.isclose(np.dot(x, y), 0.0, atol=atol)
        and np.allclose(np.cross(d, x), y, atol=atol)
    )


def test_prune_colinear():
    arr = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
    pruned = geometry.prune_colinear(arr, colinear_threshold=0.9999)
    expected = np.array([[0, 0, 0], [3, 0, 0]], dtype=float)
    npt.assert_array_equal(pruned, expected)

    # Edge cases: 0, 1, and 2 points
    empty_arr = np.empty((0, 3), dtype=float)
    npt.assert_array_equal(geometry.prune_colinear(empty_arr), empty_arr)

    single_point = np.array([[1.0, 2.0, 3.0]])
    npt.assert_array_equal(geometry.prune_colinear(single_point), single_point)

    two_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    npt.assert_array_equal(geometry.prune_colinear(two_points), two_points)

    # Sequence inputs (list and tuple)
    list_pts = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]
    npt.assert_array_equal(geometry.prune_colinear(list_pts), expected)

    tuple_pts = ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0))
    npt.assert_array_equal(geometry.prune_colinear(tuple_pts), expected)

    assert geometry.prune_colinear([]).shape == (0, 3)
    assert geometry.prune_colinear(()).shape == (0, 3)

    # Invalid input types
    with pytest.raises(ValueError, match=r"arr must be a list, tuple, or ndarray"):
        geometry.prune_colinear("invalid")

    with pytest.raises(ValueError, match=r"arr must be a list, tuple, or ndarray"):
        geometry.prune_colinear(123)

    # Invalid array dimensions
    with pytest.raises(ValueError, match=r"arr must have shape \(N, 3\)"):
        geometry.prune_colinear([1, 2, 3])

    with pytest.raises(ValueError, match=r"arr must have shape \(N, 3\)"):
        geometry.prune_colinear([[1, 2], [3, 4]])

    with pytest.raises(ValueError, match=r"arr must have shape \(N, 3\)"):
        geometry.prune_colinear(np.ones((2, 2, 3)))


def test_axes_for_dir():
    d = np.array([1, 1, 1], dtype=float) / np.sqrt(3)
    x, y = geometry.axes_for_dir(d.copy())
    assert _is_orthonormal(d, x, y)

    # When prev_x is provided and orthogonal / non-parallel to d
    prev_x = np.array([1.0, 0.0, 0.0])
    d2 = np.array([0.0, 0.0, 1.0])
    x, y = geometry.axes_for_dir(d2, prev_x=prev_x)
    assert _is_orthonormal(d2, x, y)
    npt.assert_allclose(x, [1.0, 0.0, 0.0], atol=1e-6)
    npt.assert_allclose(y, [0.0, 1.0, 0.0], atol=1e-6)

    # When prev_x is parallel (or antiparallel) or zero vector
    for parallel_prev_x in [d2, -d2, 2.0 * d2, np.zeros(3)]:
        x, y = geometry.axes_for_dir(d2, prev_x=parallel_prev_x)
        assert _is_orthonormal(d2, x, y)
        assert not np.isnan(x).any()
        assert not np.isnan(y).any()

    # When prev_x contains NaN
    x_default, y_default = geometry.axes_for_dir(d2, prev_x=None)
    for nan_prev_x in [
        np.array([np.nan, 0.0, 1.0]),
        np.full(3, np.nan),
        [np.nan, 1.0, 0.0],
    ]:
        x, y = geometry.axes_for_dir(d2, prev_x=nan_prev_x)
        assert _is_orthonormal(d2, x, y)
        assert not np.isnan(x).any()
        assert not np.isnan(y).any()
        npt.assert_allclose(x, x_default)
        npt.assert_allclose(y, y_default)


def test_rotate_vector():
    v = np.array([1.0, 0.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])
    angle = np.pi / 2
    expected = np.array([0.0, 1.0, 0.0])
    rotated = geometry.rotate_vector(v, axis, angle)
    npt.assert_allclose(rotated, expected, atol=1e-6)

    rotated_self = geometry.rotate_vector(v, v, np.pi / 3)
    npt.assert_allclose(
        rotated_self / np.linalg.norm(rotated_self), v / np.linalg.norm(v), atol=1e-6
    )
