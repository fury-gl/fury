import re

import numpy as np
import pytest

from fury import actor
from fury.actor import Group
from fury.actor.utils import get_slices, set_group_visibility, show_slices
from fury.material import (
    VectorFieldArrowMaterial,
    VectorFieldLineMaterial,
    VectorFieldThinLineMaterial,
)


def test_valid_3d_data():
    """Test valid 3D input with default parameters."""
    data = np.random.rand(10, 20, 30)
    slicer_obj = actor.data_slicer(data)

    # Verify object type and visibility
    assert isinstance(slicer_obj, Group)
    assert slicer_obj.visible
    assert len(slicer_obj.children) == 3
    assert all(child.visible for child in slicer_obj.children)


@pytest.mark.parametrize("channels", [3, 4])
def test_valid_4d_data(channels):
    """Test valid 4D RGB/RGBA input."""
    shape = (4, 5, 6, channels)
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    slicer_obj = actor.data_slicer(data)

    assert isinstance(slicer_obj, Group)
    assert len(slicer_obj.children) == 3
    assert all(child.visible for child in slicer_obj.children)

    expected_slices = np.array([shape[0] // 2, shape[1] // 2, shape[2] // 2]) + 1e-3
    np.testing.assert_array_equal(
        get_slices(slicer_obj), expected_slices.astype(np.float32)
    )

    swapped_shape = (shape[2], shape[1], shape[0], channels)
    expected_range = (float(data.min()), float(data.max()))
    for child in slicer_obj.children:
        assert child.geometry.grid.data.shape == swapped_shape
        assert child.geometry.grid.data.dtype == np.float32
        np.testing.assert_allclose(child.material.clim, expected_range)


def test_invalid_4d_data_with_incorrect_color_channel():
    """Test invalid 4D data shape."""
    data = np.random.rand(10, 20, 30, 5)  # Last dim ≠ 3 or 4
    with pytest.raises(ValueError) as excinfo:
        actor.data_slicer(data)
    assert "Last dimension must be of size 3 or 4." in str(excinfo.value)


def test_opacity_validation():
    """Test opacity validation raises errors for out-of-bounds values"""
    data = np.random.rand(10, 20, 30)

    # Test valid values
    for valid_opacity in [0, 0.5, 1]:
        slicer_obj = actor.data_slicer(data, opacity=valid_opacity)
        for child in slicer_obj.children:
            assert child.material.opacity == valid_opacity

    # Test invalid values
    for invalid_opacity in [-0.1, 1.1, 2.0]:
        with pytest.raises(ValueError) as excinfo:
            actor.data_slicer(data, opacity=invalid_opacity)
        assert "Opacity must be between 0 and 1" in str(excinfo.value)


def test_custom_initial_slices():
    """Test custom initial slice positions (Test Case 10)."""
    data = np.random.rand(10, 20, 30)
    slicer_obj = actor.data_slicer(data, initial_slices=(5, 10, 15))

    # Verify slice positions match input
    slices = np.asarray([5, 10, 15]) + 1e-3
    assert np.array_equal(get_slices(slicer_obj), slices.astype(np.float32))

    # Verify positions update correctly
    show_slices(slicer_obj, (2, 4, 6))
    slices = np.asarray([2, 4, 6]) + 1e-3
    assert np.array_equal(get_slices(slicer_obj), slices.astype(np.float32))


def test_visibility_control():
    """Test visibility settings through methods (Test Case 13)."""
    data = np.random.rand(10, 20, 30)
    slicer_obj = actor.data_slicer(data, visibility=(True, True, True))

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

    # Test setting cross section
    # cross section will not work without providing visibility.
    new_cross = [1, 2, 3]
    vf.cross_section = new_cross
    assert vf.visibility is None

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
    assert np.all(vf.visibility == np.asarray((True, False, True)))

    # Set cross section with visibility
    vf.cross_section = [1, 2, 3]
    # The y dimension should be -1 because visibility[1] is False
    assert np.all(vf.cross_section == np.array([1, 2, 3]))


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
    assert np.all(vf.cross_section == np.array([2, 2, 2]))
    assert np.all(vf.visibility == np.asarray((True, False, True)))


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
    assert vf.material.thickness == 0.0  # Replace with your module


def test_sph_glyph_input_validation():
    """sph_glyph: Test invalid inputs raise appropriate errors."""
    # Invalid coeffs type/dimensions
    with pytest.raises(TypeError):
        actor.sph_glyph([1, 2, 3])  # Not a numpy array
    with pytest.raises(ValueError):
        actor.sph_glyph(np.random.rand(3, 3))  # Not 4D

    # Invalid sphere specification
    with pytest.raises(TypeError):
        actor.sph_glyph(np.random.rand(2, 2, 2, 5), sphere=1.5)
    with pytest.raises(TypeError):
        actor.sph_glyph(np.random.rand(2, 2, 2, 5), sphere=("a", "b"))


def test_sph_glyph_default_behavior():
    """sph_glyph: Test function with minimal valid inputs."""
    coeffs = np.random.rand(2, 2, 2, 9)
    glyph = actor.sph_glyph(coeffs)

    assert glyph is not None
    assert isinstance(glyph, actor.SphGlyph)
    assert glyph.sphere.shape[0] == 362  # Default sphere has 362 vertices
    assert glyph.color_type == 0  # Converted for shader compatibility


def test_sph_glyph_custom_sphere():
    """sph_glyph: Test custom sphere specifications."""
    coeffs = np.random.rand(2, 2, 2, 9)

    # Named sphere
    glyph = actor.sph_glyph(coeffs, sphere="symmetric724")
    assert glyph.sphere.shape[0] == 724

    # Custom sphere
    glyph = actor.sph_glyph(coeffs, sphere=(36, 72))
    assert hasattr(glyph, "indices")


def test_sph_glyph_parameter_combinations():
    """sph_glyph: Test all valid basis_type and color_type combinations."""
    coeffs = np.random.rand(2, 2, 2, 16)

    for basis in ["standard", "descoteaux07"]:
        for idx, color in enumerate(["sign", "orientation"]):
            glyph = actor.sph_glyph(coeffs, basis_type=basis, color_type=color)
            assert glyph.color_type == idx


def test_sph_glyph_shininess_values():
    """sph_glyph: Test valid shininess values."""
    coeffs = np.random.rand(2, 2, 2, 4)

    for shininess in [0, 50, 100, 150.5]:
        glyph = actor.sph_glyph(coeffs, shininess=shininess)
        assert glyph.material.shininess == shininess


def test_SphGlyph_input_validation_coeffs():
    """SphGlyph: Test invalid coeffs inputs raise appropriate errors."""
    valid_sphere = (np.random.rand(100, 3), np.random.randint(0, 100, (50, 3)))

    # Not a numpy array
    with pytest.raises(TypeError):
        actor.SphGlyph([1, 2, 3], sphere=valid_sphere)

    # Not 4D
    with pytest.raises(ValueError):
        actor.SphGlyph(np.random.rand(3, 3), sphere=valid_sphere)

    # Empty last dimension
    with pytest.raises(ValueError):
        actor.SphGlyph(np.random.rand(2, 2, 2, 0), sphere=valid_sphere)


def test_SphGlyph_input_validation_sphere():
    """SphGlyph: Test invalid sphere inputs raise appropriate errors."""
    valid_coeffs = np.random.rand(2, 2, 2, 9)

    # Not a tuple
    with pytest.raises(TypeError):
        actor.SphGlyph(valid_coeffs, sphere=[1, 2, 3])

    # Wrong tuple length
    with pytest.raises(TypeError):
        actor.SphGlyph(valid_coeffs, sphere=(np.random.rand(100, 3),))

    # Invalid contents
    with pytest.raises(TypeError):
        actor.SphGlyph(valid_coeffs, sphere=([1, 2, 3], [4, 5, 6]))


def test_SphGlyph_initialization_defaults():
    """SphGlyph: Test initialization with default parameters."""
    coeffs = np.random.rand(2, 2, 2, 9)
    sphere = (np.random.rand(100, 3), np.random.randint(0, 100, (50, 3)))
    glyph = actor.SphGlyph(coeffs, sphere=sphere)

    assert glyph.n_coeff == 9
    assert glyph.data_shape == (2, 2, 2)
    assert glyph.color_type == 0  # Default 'sign'
    assert glyph.vertices_per_glyph == 100
    assert glyph.faces_per_glyph == 50


def test_SphGlyph_parameter_combinations():
    """SphGlyph: Test different basis_type and color_type combinations."""
    coeffs = np.random.rand(2, 2, 2, 16)
    sphere = (np.random.rand(100, 3), np.random.randint(0, 100, (50, 3)))

    # Test basis types
    for basis in ["standard", "descoteaux07"]:
        glyph = actor.SphGlyph(coeffs, sphere=sphere, basis_type=basis)
        assert hasattr(glyph.material, "n_coeffs")

    # Test color types
    glyph_sign = actor.SphGlyph(coeffs, sphere=sphere, color_type="sign")
    assert glyph_sign.color_type == 0

    glyph_orient = actor.SphGlyph(coeffs, sphere=sphere, color_type="orientation")
    assert glyph_orient.color_type == 1


def test_SphGlyph_shininess_values():
    """SphGlyph: Test different shininess values."""
    coeffs = np.random.rand(2, 2, 2, 4)
    sphere = (np.random.rand(100, 3), np.random.randint(0, 100, (50, 3)))

    for shininess in [0, 50, 100, 150.5]:
        glyph = actor.SphGlyph(coeffs, sphere=sphere, shininess=shininess)
        assert glyph.material.shininess == shininess


def test_SphGlyph_geometry_properties():
    """SphGlyph: Test geometry properties are correctly set."""
    coeffs = np.random.rand(3, 3, 3, 9)
    vertices = np.random.rand(200, 3)
    faces = np.random.randint(0, 200, (100, 3))
    sphere = (vertices, faces)

    glyph = actor.SphGlyph(coeffs, sphere=sphere)

    # Check positions scaling
    assert glyph.geometry.positions.data.shape[0] == 3 * 3 * 3 * 200
    assert glyph.geometry.indices.data.shape[0] == 3 * 3 * 3 * 100

    # Check SH coefficients
    assert glyph.sh_coeff.shape[0] == 3 * 3 * 3 * 9
    assert glyph.sf_func.shape[0] == 200 * glyph.material.n_coeffs
