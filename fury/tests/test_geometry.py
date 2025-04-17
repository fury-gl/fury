import numpy as np
import numpy.testing as npt

from fury import geometry, material
from fury.lib import Text


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


def test_create_mesh():
    positions = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]).astype("float32")
    geo = geometry.buffer_to_geometry(positions)
    mat = material._create_mesh_material(
        material="phong", color=(1, 0, 0), opacity=0.5, mode="auto"
    )
    mesh = geometry.create_mesh(geometry=geo, material=mat)
    assert mesh.geometry == geo
    assert mesh.material == mat


def test_line_buffer_separator():
    line_vertices = [
        np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
        np.array([[2, 2, 2], [3, 3, 3]], dtype=np.float32),
    ]
    positions, colors = geometry.line_buffer_separator(line_vertices)
    # Check positions
    npt.assert_array_equal(positions[:2], line_vertices[0])
    assert np.all(np.isnan(positions[2]))
    npt.assert_array_equal(positions[3:], line_vertices[1])
    assert colors is None

    line_vertices = [
        np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
        np.array([[2, 2, 2], [3, 3, 3]], dtype=np.float32),
    ]
    color = np.array([[1.0, 0, 0], [0.0, 1.0, 0.0]], dtype=np.float32)
    positions, colors = geometry.line_buffer_separator(
        line_vertices, color=color, color_mode="line"
    )
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
    positions, colors = geometry.line_buffer_separator(
        line_vertices, color=color, color_mode="vertex_flattened"
    )
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
    color = np.array([1, 0, 0], dtype=np.float32)  # Invalid shape
    with npt.assert_raises(ValueError):
        geometry.line_buffer_separator(line_vertices, color=color, color_mode="auto")

    line_vertices = [np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)]
    positions, colors = geometry.line_buffer_separator(line_vertices)
    npt.assert_array_equal(positions, line_vertices[0])
    assert colors is None

    line_vertices = [np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)]
    color = np.array([[1, 0, 0]], dtype=np.float32)
    positions, colors = geometry.line_buffer_separator(
        line_vertices, color=color, color_mode="line"
    )
    expected_colors = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)
    npt.assert_array_equal(colors, expected_colors)


def test_create_point():
    vertices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]).astype("float32")
    geo = geometry.buffer_to_geometry(vertices)
    mat = material._create_points_material(
        material="basic", color=(1, 0, 0), opacity=0.5, mode="auto"
    )
    point = geometry.create_point(geometry=geo, material=mat)
    assert point.geometry == geo
    assert point.material == mat


def test_create_text():
    text = "FURY"
    mat = material._create_text_material(color=(1, 0, 0), opacity=0.5)
    text_obj = geometry.create_text(text=text, material=mat)
    assert text_obj.material == mat
    assert isinstance(text_obj, Text)


def test_create_image():
    img_data = np.random.rand(128, 128)
    mat = material._create_image_material()
    image_obj = geometry.create_image(image_input=img_data, material=mat)
    assert image_obj.material == mat
    assert isinstance(image_obj, geometry.Image)
    assert image_obj.geometry.grid.dim == 2
