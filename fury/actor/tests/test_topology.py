import numpy as np
import pytest

from fury.actor import Group, contour_from_volume, surface
from fury.actor.tests._helpers import random_png
from fury.io import load_image_texture
from fury.lib import MeshBasicMaterial, MeshPhongMaterial, TextureMap
from fury.utils import generate_planar_uvs


def test_surface_basic_vertices_and_faces():
    """Test surface creation with basic vertices and faces."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    surface_actor = surface(vertices, faces)

    assert np.array_equal(surface_actor.geometry.positions.data, vertices)
    assert np.array_equal(surface_actor.geometry.indices.data, faces)
    assert not hasattr(surface_actor.geometry, "texcoords")
    assert not hasattr(surface_actor.geometry, "colors")
    assert isinstance(surface_actor.material, MeshPhongMaterial)
    assert surface_actor.material.opacity == 1.0


def test_surface_with_vertex_colors():
    """Test surface creation with vertex colors."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    surface_actor = surface(vertices, faces, colors=colors)

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

    surface_actor = surface(vertices, faces, colors=colors, normals=normals)

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

    surface_actor = surface(
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
    mesh = surface(
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
        surface(
            vertices=vertices,
            faces=faces,
            texture=str(texture_file),
            texture_coords=bad_coords,
        )

    # Test wrong dtype
    with pytest.raises(ValueError):
        bad_coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        surface(
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
        surface(vertices, faces, colors=[1, 0, 0, 1, 0])  # Invalid length

    # Test non-existent texture file
    with pytest.raises(FileNotFoundError):
        surface(vertices, faces, texture="nonexistent.png")
    # Test invalid opacity
    with pytest.raises(ValueError):
        surface(vertices, faces, opacity=1.5)  # Assuming validate_opacity raises for >1


@pytest.mark.parametrize(
    "color,opacity,material_type",
    [
        ((1, 0, 0), 0.5, MeshPhongMaterial),
        ((0, 1, 0), 0.8, MeshBasicMaterial),
    ],
)
def test_contour_from_volume(color, opacity, material_type):
    """Test contour_from_volume with various parameters."""
    data = np.zeros((3, 3, 3), dtype=int)
    data[1, 1, 1] = 1

    contours = contour_from_volume(
        data,
        color=color,
        opacity=opacity,
        material="phong" if material_type == MeshPhongMaterial else "basic",
    )

    assert isinstance(contours, Group)
    assert len(contours.children) > 0
    actor = contours.children[0]
    assert np.allclose(actor.material.color[:3], color)
    assert actor.material.opacity == pytest.approx(opacity)
    assert isinstance(actor.material, material_type)


def test_contour_from_volume_invalid_color():
    """Test contour_from_volume with invalid color."""
    data = np.zeros((3, 3, 3), dtype=int)
    data[1, 1, 1] = 1

    with pytest.raises(ValueError, match="must have 3 or 4 channels"):
        contour_from_volume(data, color=(1, 0))
