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
