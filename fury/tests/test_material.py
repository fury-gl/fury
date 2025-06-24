import numpy as np
import pytest

from fury import material, window
from fury.geometry import buffer_to_geometry, create_mesh
from fury.lib import (
    ImageBasicMaterial,
    LineArrowMaterial,
    LineMaterial,
    LineSegmentMaterial,
    LineThinMaterial,
    LineThinSegmentMaterial,
    PointsGaussianBlobMaterial,
    PointsMarkerMaterial,
    PointsMaterial,
    TextMaterial,
    Texture,
    TextureMap,
)
from fury.material import (
    SphGlyphMaterial,
    VectorFieldArrowMaterial,
    VectorFieldLineMaterial,
    VectorFieldThinLineMaterial,
    _create_mesh_material,
    _create_vector_field_material,
)
from fury.primitive import prim_sphere


def test_create_mesh_material():
    color = (1, 0, 0)
    mat = material._create_mesh_material(
        material="phong", color=color, opacity=0.5, mode="auto"
    )
    assert isinstance(mat, material.MeshPhongMaterial)
    assert mat.color == color + (0.5,)
    assert mat.color_mode == "auto"

    color = (1, 0, 0, 0.5)
    mat = material._create_mesh_material(
        material="phong", color=color, opacity=0.5, mode="auto", flat_shading=False
    )
    assert isinstance(mat, material.MeshPhongMaterial)
    assert mat.color == (1, 0, 0, 0.25)
    assert mat.color_mode == "auto"
    assert mat.flat_shading is False

    color = (1, 0, 0)
    mat = material._create_mesh_material(
        material="phong", color=color, opacity=0.5, mode="vertex"
    )
    assert isinstance(mat, material.MeshPhongMaterial)
    assert mat.color == (1, 1, 1)
    assert mat.color_mode == "vertex"

    color = (1, 0, 0)
    mat = material._create_mesh_material(
        material="basic",
        color=color,
        mode="vertex",
        enable_picking=False,
        flat_shading=True,
    )
    assert isinstance(mat, material.MeshBasicMaterial)
    assert mat.color == (1, 1, 1)
    assert mat.color_mode == "vertex"
    assert mat.flat_shading is True

    verts, faces = prim_sphere()

    geo = buffer_to_geometry(
        indices=faces.astype("int32"),
        positions=verts.astype("float32"),
        texcoords=verts.astype("float32"),
        colors=np.ones_like(verts).astype("float32"),
    )

    mat = _create_mesh_material(
        material="phong", enable_picking=False, flat_shading=False
    )

    obj = create_mesh(geometry=geo, material=mat)

    scene = window.Scene()

    scene.add(obj)

    # window.snapshot(scene=scene, fname="mat_test_1.png")
    #
    # img = Image.open("mat_test_1.png")
    # img_array = np.array(img)
    #
    # mean_r, mean_g, mean_b, _ = np.mean(
    #     img_array.reshape(-1, img_array.shape[2]), axis=0
    # )
    #
    # assert 0 <= mean_r <= 255 and 0 <= mean_g <= 255 and 0 <= mean_b <= 255
    #
    # assert sum([mean_r, mean_g, mean_b]) > 0


def test_create_point_material():
    color = (1, 0, 0)
    mat = material._create_points_material(
        material="basic", color=color, opacity=0.5, mode="auto"
    )
    assert isinstance(mat, PointsMaterial)
    assert mat.color == color + (0.5,)
    assert mat.color_mode == "auto"

    color = (1, 0, 0)
    mat = material._create_points_material(
        material="gaussian", color=color, opacity=0.5, mode="auto"
    )
    assert isinstance(mat, PointsGaussianBlobMaterial)
    assert mat.color == color + (0.5,)
    assert mat.color_mode == "auto"

    color = (1, 0, 0)
    mat = material._create_points_material(
        material="marker", color=color, opacity=0.5, mode="auto"
    )
    assert isinstance(mat, PointsMarkerMaterial)
    assert mat.color == color + (0.5,)
    assert mat.color_mode == "auto"

    color = (1, 0, 0, 0.5)
    mat = material._create_points_material(
        material="basic", color=color, opacity=0.5, mode="auto"
    )
    assert isinstance(mat, PointsMaterial)
    assert mat.color == (1, 0, 0, 0.25)
    assert mat.color_mode == "auto"

    color = (1, 0, 0)
    mat = material._create_points_material(
        material="basic", color=color, opacity=0.5, mode="vertex"
    )
    assert isinstance(mat, PointsMaterial)
    assert mat.color == (1, 1, 1)
    assert mat.color_mode == "vertex"


def test_create_text_material():
    color = (1, 0, 0)
    mat = material._create_text_material(color=color, opacity=0.5)
    assert isinstance(mat, TextMaterial)
    assert mat.color == color + (0.5,)
    assert mat.outline_color == (0, 0, 0)
    assert mat.outline_thickness == 0.0
    assert mat.weight_offset == 1.0
    assert mat.aa is True

    color = (1, 0, 0, 0.5)
    mat = material._create_text_material(color=color, opacity=0.5)
    assert isinstance(mat, TextMaterial)
    assert mat.color == (1, 0, 0, 0.25)
    assert mat.outline_color == (0, 0, 0)
    assert mat.outline_thickness == 0.0
    assert mat.weight_offset == 1.0
    assert mat.aa is True


def test_create_line_material():
    color = (1, 0, 0)
    mat = material._create_line_material(
        material="basic", color=color, opacity=0.5, mode="auto"
    )
    assert isinstance(mat, LineMaterial)
    assert mat.color == color + (0.5,)
    assert mat.color_mode == "auto"

    color = (1, 0, 0)
    mat = material._create_line_material(
        material="arrow", color=color, opacity=0.5, mode="auto"
    )
    assert isinstance(mat, LineArrowMaterial)
    assert mat.color == color + (0.5,)
    assert mat.color_mode == "auto"

    color = (1, 0, 0)
    mat = material._create_line_material(
        material="thin", color=color, opacity=0.5, mode="auto"
    )
    assert isinstance(mat, LineThinMaterial)
    assert mat.color == color + (0.5,)
    assert mat.color_mode == "auto"

    color = (1, 0, 0)
    mat = material._create_line_material(
        material="thin_segment", color=color, opacity=0.5, mode="auto"
    )
    assert isinstance(mat, LineThinSegmentMaterial)
    assert mat.color == color + (0.5,)
    assert mat.color_mode == "auto"

    color = (1, 0, 0)
    mat = material._create_line_material(
        material="segment", color=color, opacity=0.5, mode="auto"
    )
    assert isinstance(mat, LineSegmentMaterial)
    assert mat.color == color + (0.5,)
    assert mat.color_mode == "auto"


def test_create_image_material():
    mat = material._create_image_material()
    assert isinstance(mat, ImageBasicMaterial)
    assert mat.clim == (0.0, 1.0)
    assert mat.map is None
    assert mat.gamma == 1.0
    assert mat.interpolation == "nearest"

    texture = Texture(np.random.rand(256, 256).astype(np.float32), dim=2)
    mat = material._create_image_material(
        map=TextureMap(texture), gamma=0.5, clim=(0.2, 0.8), interpolation="linear"
    )
    assert isinstance(mat, ImageBasicMaterial)
    assert isinstance(mat.map, TextureMap)
    assert mat.gamma == 0.5
    assert mat.interpolation == "linear"
    assert np.allclose(mat.clim, (0.2, 0.8))


def test_VectorFieldThinLineMaterial_initialization():
    """Test VectorFieldThinLineMaterial initialization with valid cross_section."""
    cross_section = [1, 2, 3]
    material = VectorFieldThinLineMaterial(cross_section)
    assert np.array_equal(material.cross_section, cross_section)
    assert material.color_mode == "vertex"


def test_VectorFieldThinLineMaterial_cross_section_property():
    """Test VectorFieldThinLineMaterial cross_section property getter and setter."""
    cross_section = [1, 2, 3]
    material = VectorFieldThinLineMaterial(cross_section)

    # Test getter
    assert np.array_equal(material.cross_section, cross_section)

    # Test setter with valid input
    new_cross_section = [4, 5, 6]
    material.cross_section = new_cross_section
    assert np.array_equal(material.cross_section, new_cross_section)

    # Test setter with numpy array
    np_cross_section = np.array([7, 8, 9], dtype=np.int32)
    material.cross_section = np_cross_section
    assert np.array_equal(material.cross_section, np_cross_section)


def test_VectorFieldThinLineMaterial_invalid_cross_section():
    """Test VectorFieldThinLineMaterial with invalid cross_section inputs."""
    material = VectorFieldThinLineMaterial([0, 0, 0])

    # Test wrong length
    with pytest.raises(
        ValueError, match="cross_section must have exactly 3 dimensions"
    ):
        material.cross_section = [1, 2]

    # Test non-integer values
    with pytest.raises(ValueError, match="cross_section must contain only integers"):
        material.cross_section = [1.5, 2.0, 3.0]

    # Test invalid types
    with pytest.raises(ValueError):
        material.cross_section = "invalid"


def test_VectorFieldThinLineMaterial_uniform_buffer_update():
    """Test VectorFieldThinLineMaterial updates uniform buffer correctly."""
    cross_section = [1, 2, 3]
    material = VectorFieldThinLineMaterial(cross_section)

    # Check uniform buffer contains correct data
    uniform_data = material.uniform_buffer.data["cross_section"]
    assert np.array_equal(uniform_data[:3], cross_section)
    assert uniform_data[3] == 0  # padding value

    # Update and verify
    new_cross_section = [4, 5, 6]
    material.cross_section = new_cross_section
    updated_data = material.uniform_buffer.data["cross_section"]
    assert np.array_equal(updated_data[:3], new_cross_section)


def test_VectorFieldLineMaterial_inheritance():
    """Test VectorFieldLineMaterial inherits properly from
    VectorFieldThinLineMaterial.
    """
    cross_section = [1, 2, 3]
    material = VectorFieldLineMaterial(cross_section)

    # Verify inheritance
    assert isinstance(material, VectorFieldThinLineMaterial)
    assert np.array_equal(material.cross_section, cross_section)
    assert material.color_mode == "vertex"


def test_VectorFieldArrowMaterial_inheritance():
    """Test VectorFieldArrowMaterial inherits properly from
    VectorFieldThinLineMaterial.
    """
    cross_section = [1, 2, 3]
    material = VectorFieldArrowMaterial(cross_section)

    # Verify inheritance
    assert isinstance(material, VectorFieldThinLineMaterial)
    assert np.array_equal(material.cross_section, cross_section)
    assert material.color_mode == "vertex"


def test_VectorFieldThinLineMaterial_with_numpy_inputs():
    """Test VectorFieldThinLineMaterial works with numpy array inputs."""
    # Test initialization with numpy array
    np_cross_section = np.array([1, 2, 3], dtype=np.int32)
    material = VectorFieldThinLineMaterial(np_cross_section)
    assert np.array_equal(material.cross_section, np_cross_section)

    # Test setting with numpy array
    new_np_cross_section = np.array([4, 5, 6], dtype=np.int64)
    material.cross_section = new_np_cross_section
    assert np.array_equal(material.cross_section, new_np_cross_section)


def test_VectorFieldThinLineMaterial_edge_cases():
    """Test VectorFieldThinLineMaterial with edge case inputs."""
    # Test with zeros
    material = VectorFieldThinLineMaterial([0, 0, 0])
    assert np.array_equal(material.cross_section, [0, 0, 0])

    # Test with negative values
    material.cross_section = [-1, -2, -3]
    assert np.array_equal(material.cross_section, [-1, -2, -3])

    # Test with large values
    large_values = [999999, 999999, 999999]
    material.cross_section = large_values
    assert np.array_equal(material.cross_section, large_values)


def test_create_vector_field_material_thin_line():
    """Test creating a thin line material with default parameters."""
    cross_section = [1, 2, 3]
    material = _create_vector_field_material(cross_section, material="thin_line")

    assert isinstance(material, VectorFieldThinLineMaterial)
    assert material.pick_write is True
    assert material.opacity == 1.0
    assert material.thickness == 1.0
    assert material.thickness_space == "screen"
    assert material.aa is True
    assert np.array_equal(material.cross_section, cross_section)


def test_create_vector_field_material_line():
    """Test creating a line material with custom parameters."""
    cross_section = [4, 5, 6]
    material = _create_vector_field_material(
        cross_section,
        material="line",
        enable_picking=False,
        opacity=0.5,
        thickness=2.0,
        thickness_space="world",
        anti_aliasing=False,
    )

    assert isinstance(material, VectorFieldLineMaterial)
    assert material.pick_write is False
    assert material.opacity == 0.5
    assert material.thickness == 2.0
    assert material.thickness_space == "world"
    assert material.aa is False
    assert np.array_equal(material.cross_section, cross_section)


def test_create_vector_field_material_arrow():
    """Test creating an arrow material with custom parameters."""
    cross_section = [7, 8, 9]
    material = _create_vector_field_material(
        cross_section,
        material="arrow",
        opacity=0.8,
        thickness=1.5,
    )

    assert isinstance(material, VectorFieldArrowMaterial)
    assert material.pick_write is True  # default
    assert np.allclose(material.opacity, 0.8, rtol=1e-2)
    assert material.thickness == 1.5
    assert material.thickness_space == "screen"  # default
    assert material.aa is True  # default
    assert np.array_equal(material.cross_section, cross_section)


def test_create_vector_field_material_invalid_type():
    """Test creating a material with invalid type raises error."""
    cross_section = [1, 1, 1]
    with pytest.raises(ValueError, match="Unsupported material type: invalid"):
        _create_vector_field_material(cross_section, material="invalid")


def test_create_vector_field_material_edge_cases():
    """Test edge cases for material creation."""
    # Test with zero opacity
    material = _create_vector_field_material([1, 1, 1], opacity=0.0)
    assert material.opacity == 0.0

    # Test with zero thickness
    material = _create_vector_field_material([1, 1, 1], thickness=0.0)
    assert material.thickness == 0.0

    # Test with negative thickness (should not work)
    material = _create_vector_field_material([1, 1, 1], thickness=-1.0)
    assert material.thickness == 0.0


def test_create_vector_field_material_thickness_spaces():
    """Test different thickness space options."""
    for space in ["screen", "world", "model"]:
        material = _create_vector_field_material(
            [1, 1, 1],
            thickness_space=space,
        )
        assert material.thickness_space == space


def test_create_vector_field_material_anti_aliasing():
    """Test anti-aliasing parameter."""
    material = _create_vector_field_material([1, 1, 1], anti_aliasing=False)
    assert material.aa is False

    material = _create_vector_field_material([1, 1, 1], anti_aliasing=True)
    assert material.aa is True


def test_create_vector_field_material_picking():
    """Test enable_picking parameter."""
    material = _create_vector_field_material([1, 1, 1], enable_picking=False)
    assert material.pick_write is False

    material = _create_vector_field_material([1, 1, 1], enable_picking=True)
    assert material.pick_write is True


def test_SphGlyphMaterial_initialization_defaults():
    """SphGlyphMaterial: Test initialization with default parameters."""
    material = SphGlyphMaterial()

    assert material.l_max == 4
    assert material.scale == 2
    assert material.shininess == 30
    assert material.emissive == "#000"
    assert material.specular == "#494949"


def test_SphGlyphMaterial_l_max_property():
    """SphGlyphMaterial: Test l_max property validation and updates."""
    material = SphGlyphMaterial()

    # Test valid even integers
    for value in [2, 4, 6, 8]:
        material.l_max = value
        assert material.l_max == value

    # Test invalid values
    with pytest.raises(ValueError):
        material.l_max = "4"  # Not integer
    with pytest.raises(ValueError):
        material.l_max = 4.5  # Float


def test_SphGlyphMaterial_scale_property():
    """SphGlyphMaterial: Test scale property validation and updates."""
    material = SphGlyphMaterial()

    # Test valid numbers
    for value in [1, 1.5, 2.0, 3.5, 10]:
        material.scale = value
        assert material.scale == value

    # Test invalid values
    with pytest.raises(ValueError):
        material.scale = "2"  # String
    with pytest.raises(ValueError):
        material.scale = None  # None


def test_SphGlyphMaterial_custom_initialization():
    """SphGlyphMaterial: Test initialization with custom parameters."""
    material = SphGlyphMaterial(
        l_max=6,
        scale=3.5,
        shininess=50,
        emissive="#111",
        specular="#888",
        flat_shading=True,
    )

    assert material.l_max == 6
    assert material.scale == 3.5
    assert material.shininess == 50
    assert material.emissive == "#111"
    assert material.specular == "#888"
    assert material.flat_shading is True


def test_SphGlyphMaterial_uniform_type():
    """SphGlyphMaterial: Test uniform_type contains expected fields."""
    assert "l_max" in SphGlyphMaterial.uniform_type
    assert "scale" in SphGlyphMaterial.uniform_type
    assert SphGlyphMaterial.uniform_type["l_max"] == "i4"
    assert SphGlyphMaterial.uniform_type["scale"] == "f4"

    # Check inheritance from MeshPhongMaterial
    assert "shininess" in SphGlyphMaterial.uniform_type
    assert "emissive_color" in SphGlyphMaterial.uniform_type
