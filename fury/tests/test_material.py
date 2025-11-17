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
    StreamlinesMaterial,
    VectorFieldArrowMaterial,
    VectorFieldLineMaterial,
    VectorFieldThinLineMaterial,
    _StreamtubeBakedMaterial,
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


def test_create_point_material():
    color = (1, 0, 0)
    mat = material._create_points_material(material="basic", color=color, mode="auto")
    assert isinstance(mat, PointsMaterial)
    assert mat.color == color
    assert mat.color_mode == "auto"

    color = (1, 0, 0)
    mat = material._create_points_material(
        material="gaussian", color=color, opacity=0.5, mode="auto"
    )
    assert isinstance(mat, PointsGaussianBlobMaterial)
    assert mat.color == color
    assert mat.color_mode == "auto"
    assert np.round(mat.opacity, 1) == 0.5

    color = (1, 0, 0)
    mat = material._create_points_material(
        material="marker", color=color, opacity=0.5, mode="auto"
    )
    assert isinstance(mat, PointsMarkerMaterial)
    assert mat.color == color
    assert mat.color_mode == "auto"

    color = (1, 0, 0, 0.5)
    mat = material._create_points_material(
        material="basic", color=color, opacity=0.5, mode="auto"
    )
    assert isinstance(mat, PointsMaterial)
    assert mat.color == (1, 0, 0, 0.5)
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


def test_visibility_none():
    """Test setting visibility to None."""
    cross_section = [7, 8, 9]
    material = _create_vector_field_material(
        cross_section,
        material="arrow",
        opacity=0.8,
        thickness=1.5,
    )

    material.visibility = None
    assert material.visibility is None  # Default behavior
    assert np.array_equal(
        material.uniform_buffer.data["visibility"],
        np.array([-1, -1, -1, 0], dtype=np.int32),  # As per implementation
    )

    material = _create_vector_field_material(
        cross_section,
        visibility=(True, False, True),
        material="arrow",
        opacity=0.8,
        thickness=1.5,
    )

    assert material.visibility == [True, False, True]  # Default behavior
    assert np.array_equal(
        material.uniform_buffer.data["visibility"],
        np.array([1, 0, 1, 0], dtype=np.int32),  # As per implementation
    )


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

    assert material.n_coeffs == -1
    assert material.scale == 1
    assert material.shininess == 30
    assert material.emissive == "#000"
    assert material.specular == "#494949"


def test_SphGlyphMaterial_n_coeffs_property():
    """SphGlyphMaterial: Test n_coeffs property validation and updates."""
    material = SphGlyphMaterial()

    # Test valid integer values
    for value in [1, 4, 9, 16, 25, 36]:
        material.n_coeffs = value
        assert material.n_coeffs == value

    # Test invalid values
    with pytest.raises(ValueError):
        material.n_coeffs = "4"  # Not integer
    with pytest.raises(ValueError):
        material.n_coeffs = 4.5  # Float


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
        n_coeffs=36,
        scale=3.5,
        shininess=50,
        emissive="#111",
        specular="#888",
        flat_shading=True,
    )

    assert material.n_coeffs == 36
    assert material.scale == 3.5
    assert material.shininess == 50
    assert material.emissive == "#111"
    assert material.specular == "#888"
    assert material.flat_shading is True


def test_SphGlyphMaterial_uniform_type():
    """SphGlyphMaterial: Test uniform_type contains expected fields."""
    assert "n_coeffs" in SphGlyphMaterial.uniform_type
    assert "scale" in SphGlyphMaterial.uniform_type
    assert SphGlyphMaterial.uniform_type["n_coeffs"] == "i4"
    assert SphGlyphMaterial.uniform_type["scale"] == "f4"

    # Check inheritance from MeshPhongMaterial
    assert "shininess" in SphGlyphMaterial.uniform_type
    assert "emissive_color" in SphGlyphMaterial.uniform_type


def test_StreamlinesMaterial_initialization_defaults():
    """StreamlinesMaterial: Test initialization with default parameters."""
    material = StreamlinesMaterial()

    assert material.outline_thickness == 0.0
    assert np.array_equal(material.outline_color, (0, 0, 0))


def test_StreamlinesMaterial_custom_initialization():
    """StreamlinesMaterial: Test initialization with custom parameters."""
    outline_thickness = 2.5
    outline_color = (1.0, 0.5, 0.2)

    material = StreamlinesMaterial(
        outline_thickness=outline_thickness,
        outline_color=outline_color,
        thickness=3.0,
        opacity=0.7,
        color=(0.8, 0.2, 0.4),
    )

    assert material.outline_thickness == outline_thickness
    # The outline_color getter returns RGB part only, so we compare with RGB tuple
    assert np.allclose(material.outline_color, outline_color)
    assert material.thickness == 3.0
    assert round(material.opacity, 2) == 0.7


def test_StreamlinesMaterial_inheritance():
    """StreamlinesMaterial: Test that it properly inherits from LineMaterial."""
    material = StreamlinesMaterial()

    # Test that it inherits LineMaterial properties
    assert hasattr(material, "thickness")
    assert hasattr(material, "color")
    assert hasattr(material, "opacity")

    # Test that parent class methods work
    material.thickness = 4.0
    assert material.thickness == 4.0

    material.opacity = 0.5
    assert material.opacity == 0.5


def test_StreamlinesMaterial_with_kwargs():
    """StreamlinesMaterial: Test initialization with additional kwargs."""
    material = StreamlinesMaterial(
        outline_thickness=1.5,
        outline_color=(0.5, 0.5, 0.5),
        # LineMaterial kwargs
        thickness=2.0,
        color=(1.0, 0.0, 0.0),
        opacity=0.8,
        thickness_space="world",
        aa=False,
    )

    # Test StreamlinesMaterial specific properties
    assert material.outline_thickness == 1.5
    assert np.allclose(material.outline_color, (0.5, 0.5, 0.5))

    # Test inherited properties
    assert material.thickness == 2.0
    assert round(material.opacity, 1) == 0.8


def test_StreamlinesMaterial_outline_thickness_property():
    """StreamlinesMaterial: Test outline_thickness property getter and setter."""
    material = StreamlinesMaterial()

    # Test default value
    assert material.outline_thickness == 0.0

    # Test setter with valid values
    for thickness in [0.5, 1.0, 2.5, 5.0]:
        material.outline_thickness = thickness
        assert material.outline_thickness == thickness

    # Test setter with zero
    material.outline_thickness = 0.0
    assert material.outline_thickness == 0.0

    # Test setter with integer (should convert to float)
    material.outline_thickness = 3
    assert material.outline_thickness == 3.0
    assert isinstance(material.outline_thickness, float)


def test_StreamlinesMaterial_outline_color_property():
    """StreamlinesMaterial: Test outline_color property getter and setter."""
    material = StreamlinesMaterial()

    # Test default value (getter returns RGB part only)
    expected_default_rgb = (0, 0, 0)
    assert np.allclose(material.outline_color, expected_default_rgb)

    # Test setter with RGB tuple
    rgb_color = (1.0, 0.5, 0.2)
    material.outline_color = rgb_color
    # Getter returns RGB part only
    assert np.allclose(material.outline_color, rgb_color)

    # Test setter with RGBA tuple
    rgba_color = (0.8, 0.3, 0.7, 0.9)
    material.outline_color = rgba_color
    # Getter returns RGB part only
    expected_rgb = rgba_color[:3]
    assert np.allclose(material.outline_color, expected_rgb)

    # Test setter with list
    list_color = [0.2, 0.8, 0.4]
    material.outline_color = list_color
    assert np.allclose(material.outline_color, list_color)

    # Test setter with numpy array
    np_color = np.array([0.6, 0.1, 0.9])
    material.outline_color = np_color
    assert np.allclose(material.outline_color, np_color)


def test_StreamlinesMaterial_uniform_type():
    """StreamlinesMaterial: Test uniform_type contains expected fields."""
    assert "outline_thickness" in StreamlinesMaterial.uniform_type
    assert "outline_color" in StreamlinesMaterial.uniform_type
    assert StreamlinesMaterial.uniform_type["outline_thickness"] == "f4"
    assert StreamlinesMaterial.uniform_type["outline_color"] == "4xf4"

    # Check inheritance from LineMaterial
    assert "thickness" in StreamlinesMaterial.uniform_type
    assert "color" in StreamlinesMaterial.uniform_type
    assert "opacity" in StreamlinesMaterial.uniform_type


def test_StreamlinesMaterial_edge_cases():
    """StreamlinesMaterial: Test edge cases and boundary conditions."""
    material = StreamlinesMaterial()

    # Test negative outline_thickness (should still work as it's just a float)
    material.outline_thickness = -1.0
    assert material.outline_thickness == -1.0

    # Test very large outline_thickness
    material.outline_thickness = 1000.0
    assert material.outline_thickness == 1000.0

    # Test outline_color with values outside [0,1] range
    material.outline_color = (1.5, -0.5, 2.0)
    expected_rgb = (1.5, -0.5, 2.0)
    assert np.allclose(material.outline_color, expected_rgb)

    # Test empty tuple/list (should fail gracefully)
    try:
        material.outline_color = ()
    except (ValueError, IndexError, TypeError):
        pass  # Expected to fail

    # Test single value (should fail gracefully)
    try:
        material.outline_color = 0.5
    except (ValueError, IndexError, TypeError):
        pass  # Expected to fail


def test_StreamtubeBakedMaterial_defaults():
    """_StreamtubeBakedMaterial: Test default initialization populates uniforms."""
    mat = _StreamtubeBakedMaterial()

    assert isinstance(mat, material.MeshPhongMaterial)
    assert mat.radius == pytest.approx(0.2)
    assert mat.segments == 8
    assert mat.end_caps is True
    assert mat.line_count == 0
    assert mat.uniform_buffer.data["tube_radius"] == pytest.approx(0.2)
    assert mat.uniform_buffer.data["tube_segments"] == 8
    assert mat.uniform_buffer.data["tube_end_caps"] == 1
    assert mat.uniform_buffer.data["line_count"] == 0


def test_StreamtubeBakedMaterial_parameter_updates():
    """_StreamtubeBakedMaterial: Test property setters update uniforms."""
    mat = _StreamtubeBakedMaterial(radius=0.4, segments=12, end_caps=False)

    assert mat.radius == pytest.approx(0.4)
    assert mat.uniform_buffer.data["tube_radius"] == pytest.approx(0.4)

    mat.radius = 0.75
    assert mat.radius == pytest.approx(0.75)
    assert mat.uniform_buffer.data["tube_radius"] == pytest.approx(0.75)

    assert mat.segments == 12
    mat.segments = 24
    assert mat.segments == 24
    assert mat.uniform_buffer.data["tube_segments"] == 24

    assert mat.end_caps is False
    mat.end_caps = True
    assert mat.end_caps is True
    assert mat.uniform_buffer.data["tube_end_caps"] == 1

    mat.line_count = 42
    assert mat.line_count == 42
    assert mat.uniform_buffer.data["line_count"] == 42


def test_StreamtubeBakedMaterial_uniform_type():
    """_StreamtubeBakedMaterial: Uniform layout extends MeshPhongMaterial."""
    uniform_type = _StreamtubeBakedMaterial.uniform_type

    assert uniform_type["tube_radius"] == "f4"
    assert uniform_type["tube_segments"] == "u4"
    assert uniform_type["tube_end_caps"] == "i4"
    assert uniform_type["line_count"] == "u4"
    assert "color" in uniform_type
    assert "shininess" in uniform_type


def test_StreamtubeBakedMaterial_setup_compute_shader():
    """_StreamtubeBakedMaterial: Compute shader config stores parameters."""
    mat = _StreamtubeBakedMaterial(radius=0.3, segments=6, end_caps=True)

    mat._setup_compute_shader(line_count=5, max_line_length=18, tube_segments=20)

    assert mat.line_count == 5
    assert mat.segments == 20
    assert mat._max_line_length == 18
