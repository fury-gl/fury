import numpy as np
import numpy.testing as npt

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

    bent = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 1, 0]], dtype=float)
    retained = geometry.prune_colinear(bent, colinear_threshold=0.9999)
    npt.assert_array_equal(retained, bent)


def test_axes_for_dir():
    d = np.array([1, 1, 1], dtype=float) / np.sqrt(3)
    x, y = geometry.axes_for_dir(d.copy())
    assert _is_orthonormal(d, x, y)


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
