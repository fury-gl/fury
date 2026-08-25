import logging

import numpy as np
import pytest

from fury.actor import Group, contour_from_volume, surface
from fury.actor.tests._helpers import PBR_MATERIAL_TYPES, random_png
from fury.io import load_image_texture
from fury.lib import (
    MeshBasicMaterial,
    MeshPhongMaterial,
    MeshPhysicalMaterial,
    MeshStandardMaterial,
    TextureMap,
)
from fury.material import DEFAULT_PBR_ROUGHNESS
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


# --------------------------------------------------------------------------
# surface(): colors
# --------------------------------------------------------------------------

# Four vertices, not three: with exactly three, surface misreads a single RGB
# colour as per-vertex data (see test_surface_single_color_with_three_vertices).
QUAD_VERTICES = np.array(
    [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
)
QUAD_FACES = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
QUAD_NORMALS = np.tile([0.0, 0.0, 1.0], (4, 1)).astype(np.float32)


def test_surface_single_color_uses_material_color():
    """A single colour is carried by the material, not a vertex buffer."""
    surf = surface(QUAD_VERTICES, QUAD_FACES, colors=(1, 0, 0))

    assert surf.material.color_mode == "auto"
    assert not hasattr(surf.geometry, "colors")
    assert surf.material.color[:3] == pytest.approx((1, 0, 0))


@pytest.mark.parametrize(
    "colors",
    [(1.0, 0.0, 0.0), (255, 0, 0), "#FF0000", [1.0, 0.0, 0.0]],
    ids=["unit-float", "uint8", "hex", "list"],
)
def test_surface_single_color_input_formats(colors):
    """Hex, [0, 255] and [0, 1] colours all normalise to the same value."""
    surf = surface(QUAD_VERTICES, QUAD_FACES, colors=colors)

    assert surf.material.color_mode == "auto"
    assert surf.material.color[:3] == pytest.approx((1, 0, 0))


def test_surface_per_vertex_rgba_colors():
    """Per-vertex RGBA is uploaded as a colour buffer in vertex mode."""
    colors = np.array(
        [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 0.5]], dtype=np.float32
    )
    surf = surface(QUAD_VERTICES, QUAD_FACES, colors=colors)

    assert surf.material.color_mode == "vertex"
    assert np.array_equal(surf.geometry.colors.data, colors)


# --------------------------------------------------------------------------
# surface(): opacity
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "colors",
    [None, (1, 0, 0), np.ones((4, 3), dtype=np.float32)],
    ids=["no-colors", "single-color", "per-vertex"],
)
def test_surface_opacity_reaches_material(colors):
    """Opacity must be applied on every colour code path, not just one."""
    surf = surface(QUAD_VERTICES, QUAD_FACES, colors=colors, opacity=0.4)
    assert surf.material.opacity == pytest.approx(0.4)


def test_surface_single_color_opacity_multiplies_alpha():
    surf = surface(QUAD_VERTICES, QUAD_FACES, colors=(1, 0, 0), opacity=0.4)
    assert surf.material.color[3] == pytest.approx(0.4)


# --------------------------------------------------------------------------
# surface(): material selection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("phong", MeshPhongMaterial),
        ("basic", MeshBasicMaterial),
        ("standard", MeshStandardMaterial),
        ("physical", MeshPhysicalMaterial),
    ],
)
def test_surface_material_selection(name, expected):
    surf = surface(QUAD_VERTICES, QUAD_FACES, colors=(1, 0, 0), material=name)
    assert type(surf.material) is expected


def test_surface_invalid_material_raises():
    with pytest.raises(ValueError, match="Unsupported material type: bogus"):
        surface(QUAD_VERTICES, QUAD_FACES, material="bogus")


# --------------------------------------------------------------------------
# surface(): textures
# --------------------------------------------------------------------------


def _tetrahedron():
    """A non-degenerate mesh, so planar UVs exist for every axis."""
    vertices = np.array(
        [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    return vertices, faces


@pytest.fixture
def texture_file(tmpdir):
    path = tmpdir.join("texture.png")
    random_png(8, 8).save(str(path), "PNG")
    return str(path)


@pytest.mark.parametrize("axis", ["xy", "yz", "xz"])
def test_surface_texture_axis_generates_matching_uvs(axis, texture_file):
    vertices, faces = _tetrahedron()
    surf = surface(vertices, faces, texture=texture_file, texture_axis=axis)

    assert np.array_equal(
        surf.geometry.texcoords.data, generate_planar_uvs(vertices, axis=axis)
    )
    assert surf.material.color_mode == "auto"


def test_surface_texture_preserves_normals(texture_file):
    surf = surface(
        QUAD_VERTICES,
        QUAD_FACES,
        texture=texture_file,
        texture_coords=np.zeros((4, 2), dtype=np.float32),
        normals=QUAD_NORMALS,
    )
    assert np.array_equal(surf.geometry.normals.data, QUAD_NORMALS)


def test_surface_texture_opacity(texture_file):
    surf = surface(
        QUAD_VERTICES,
        QUAD_FACES,
        texture=texture_file,
        texture_coords=np.zeros((4, 2), dtype=np.float32),
        opacity=0.3,
    )
    assert surf.material.opacity == pytest.approx(0.3)
    assert surf.material.map is not None


def test_surface_texture_ignored_when_colors_given(texture_file, caplog):
    with caplog.at_level(logging.WARNING):
        surf = surface(
            QUAD_VERTICES, QUAD_FACES, colors=(1, 0, 0), texture=texture_file
        )

    assert "Texture will be ignored" in caplog.text
    assert surf.material.map is None
    assert surf.material.color_mode == "auto"


def test_surface_warns_when_texture_coords_are_missing(texture_file, caplog):
    with caplog.at_level(logging.WARNING):
        surf = surface(QUAD_VERTICES, QUAD_FACES, texture=texture_file)

    assert "planar projection" in caplog.text
    assert hasattr(surf.geometry, "texcoords")


def test_surface_texture_coords_ignored_without_texture():
    """texture_coords alone does nothing; the texture branch gates on texture."""
    surf = surface(
        QUAD_VERTICES, QUAD_FACES, texture_coords=np.zeros((4, 2), dtype=np.float32)
    )
    assert not hasattr(surf.geometry, "texcoords")


def test_surface_missing_texture_file_raises():
    with pytest.raises(FileNotFoundError, match="nonexistent.png"):
        surface(QUAD_VERTICES, QUAD_FACES, texture="nonexistent.png")


# --------------------------------------------------------------------------
# surface(): known defects, pinned so they fail loudly once fixed
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="surface() drops normals when neither colors nor texture is given: "
    "the final else branch builds the geometry without them.",
)
def test_surface_keeps_normals_without_colors_or_texture():
    surf = surface(QUAD_VERTICES, QUAD_FACES, normals=QUAD_NORMALS)
    assert np.array_equal(surf.geometry.normals.data, QUAD_NORMALS)


@pytest.mark.xfail(
    strict=True,
    reason="surface() truncates a single colour to RGB via "
    "normalize_colors(...)[0][:3], so an RGBA alpha is silently discarded.",
)
def test_surface_single_rgba_color_keeps_alpha():
    surf = surface(QUAD_VERTICES, QUAD_FACES, colors=(1, 0, 0, 0.5))
    assert surf.material.color[3] == pytest.approx(0.5)


@pytest.mark.xfail(
    strict=True,
    reason="surface() re-tests colors.shape[0] == vertices.shape[0] after "
    "normalising a single colour to an array, so a 3-vertex mesh misreads an "
    "RGB colour as per-vertex data.",
)
def test_surface_single_color_with_three_vertices():
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    surf = surface(vertices, faces, colors=(1, 0, 0))
    assert surf.material.color_mode == "auto"


# --------------------------------------------------------------------------
# surface() / contour_from_volume(): PBR materials
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_material", ["standard", "physical"])
def test_surface_pbr_material_without_colors(mesh_material):
    surf = surface(
        QUAD_VERTICES,
        QUAD_FACES,
        material=mesh_material,
        material_params={"metalness": 1.0},
    )
    assert surf.material.metalness == pytest.approx(1.0)
    assert surf.material.roughness == pytest.approx(DEFAULT_PBR_ROUGHNESS)


def test_surface_pbr_material_with_per_vertex_colors():
    surf = surface(
        QUAD_VERTICES,
        QUAD_FACES,
        colors=np.ones((4, 3), dtype=np.float32),
        material="physical",
        material_params={"clearcoat": 1.0},
    )
    assert isinstance(surf.material, MeshPhysicalMaterial)
    assert surf.material.color_mode == "vertex"
    assert surf.material.clearcoat == pytest.approx(1.0)


def test_surface_pbr_material_with_single_color():
    surf = surface(
        QUAD_VERTICES,
        QUAD_FACES,
        colors=(1, 0, 0),
        material="physical",
        material_params={"clearcoat": 1.0},
    )
    assert isinstance(surf.material, MeshPhysicalMaterial)
    assert surf.material.color_mode == "auto"
    assert surf.material.clearcoat == pytest.approx(1.0)


def test_surface_rejects_pbr_params_on_phong():
    with pytest.raises(ValueError, match="not supported by material"):
        surface(
            QUAD_VERTICES,
            QUAD_FACES,
            material="phong",
            material_params={"metalness": 1.0},
        )


@pytest.mark.parametrize("mesh_material", ["standard", "physical"])
def test_contour_from_volume_supports_pbr(mesh_material):
    data = np.zeros((12, 12, 12))
    data[3:9, 3:9, 3:9] = 1
    contours = contour_from_volume(
        data, material=mesh_material, material_params={"metalness": 1.0}
    )

    assert len(contours.children) > 0
    for child in contours.children:
        assert type(child.material) is PBR_MATERIAL_TYPES[mesh_material]
        assert child.material.metalness == pytest.approx(1.0)
