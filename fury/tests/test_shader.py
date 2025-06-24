import numpy as np

from fury.actor import SphGlyph, VectorField
from fury.lib import load_wgsl
from fury.primitive import prim_sphere
from fury.shader import (
    SphGlyphComputeShader,
    VectorFieldArrowShader,
    VectorFieldComputeShader,
    VectorFieldShader,
    VectorFieldThinShader,
)


def test_VectorFieldComputeShader_initialization():
    """Test VectorFieldComputeShader initialization."""
    field = np.random.rand(5, 5, 5, 3)
    field_shape = field.shape[:-1]  # Exclude the last dimension (vector components)
    wobject = VectorField(field)
    shader = VectorFieldComputeShader(wobject)

    assert shader["num_vectors"] == wobject.vectors_per_voxel
    assert shader["data_shape"] == field_shape
    assert shader["workgroup_size"] == 64
    assert shader.type == "compute"


def test_VectorFieldComputeShader_get_render_info():
    """Test VectorFieldComputeShader.get_render_info()."""
    field = np.random.rand(5, 5, 5, 3)
    wobject = VectorField(field)
    shader = VectorFieldComputeShader(wobject)

    render_info = shader.get_render_info(wobject, {})
    assert isinstance(render_info, dict)
    assert "indices" in render_info
    assert render_info["indices"][0] > 0  # Should have at least one workgroup


def test_VectorFieldComputeShader_get_pipeline_info():
    """Test VectorFieldComputeShader.get_pipeline_info()."""
    field = np.random.rand(5, 5, 5, 3)
    wobject = VectorField(field)
    shader = VectorFieldComputeShader(wobject)

    pipeline_info = shader.get_pipeline_info(wobject, {})
    assert isinstance(pipeline_info, dict)
    assert pipeline_info == {}


def test_VectorFieldComputeShader_get_code():
    """Test VectorFieldComputeShader.get_code()."""
    field = np.random.rand(5, 5, 5, 3)
    wobject = VectorField(field)
    shader = VectorFieldComputeShader(wobject)

    code = shader.get_code()
    assert isinstance(code, str)
    assert load_wgsl("vector_field_compute.wgsl", package_name="fury.wgsl") == code


def test_VectorFieldThinShader_initialization():
    """Test VectorFieldThinShader initialization."""
    field = np.random.rand(5, 5, 5, 3)
    field_shape = field.shape[:-1]  # Exclude the last dimension (vector components)
    wobject = VectorField(field)
    shader = VectorFieldThinShader(wobject)

    assert shader["num_vectors"] == wobject.vectors_per_voxel
    assert shader["data_shape"] == field_shape


def test_VectorFieldThinShader_get_code():
    """Test VectorFieldThinShader.get_code()."""
    field = np.random.rand(5, 5, 5, 3)
    wobject = VectorField(field)
    shader = VectorFieldThinShader(wobject)

    code = shader.get_code()
    assert isinstance(code, str)
    assert load_wgsl("vector_field_thin_render.wgsl", package_name="fury.wgsl") == code


def test_VectorFieldShader_initialization():
    """Test VectorFieldShader initialization."""
    field = np.random.rand(5, 5, 5, 3)
    field_shape = field.shape[:-1]  # Exclude the last dimension (vector components)
    wobject = VectorField(field)
    shader = VectorFieldShader(wobject)

    assert shader["num_vectors"] == wobject.vectors_per_voxel
    assert shader["data_shape"] == field_shape
    assert shader["line_type"] == "segment"


def test_VectorFieldShader_get_code():
    """Test VectorFieldShader.get_code()."""
    field = np.random.rand(5, 5, 5, 3)
    wobject = VectorField(field)
    shader = VectorFieldShader(wobject)

    code = shader.get_code()
    assert isinstance(code, str)
    assert load_wgsl("vector_field_render.wgsl", package_name="fury.wgsl") == code


def test_VectorFieldArrowShader_initialization():
    """Test VectorFieldArrowShader initialization."""
    field = np.random.rand(5, 5, 5, 3)
    field_shape = field.shape[:-1]  # Exclude the last dimension (vector components)
    wobject = VectorField(field)
    shader = VectorFieldArrowShader(wobject)

    assert shader["num_vectors"] == wobject.vectors_per_voxel
    assert shader["data_shape"] == field_shape
    assert shader["line_type"] == "arrow"


def test_VectorFieldArrowShader_inheritance():
    """Test VectorFieldArrowShader inheritance."""
    field = np.random.rand(5, 5, 5, 3)
    wobject = VectorField(field)
    shader = VectorFieldArrowShader(wobject)

    assert isinstance(shader, VectorFieldShader)
    assert hasattr(shader, "get_code")
    assert shader["line_type"] == "arrow"


def test_shaders_with_multiple_vectors_per_voxel():
    """Test shaders with multiple vectors per voxel."""
    field = np.random.rand(5, 5, 5, 10, 3)
    vectors_per_voxel = 10
    wobject = VectorField(field)

    # Test compute shader
    compute_shader = VectorFieldComputeShader(wobject)
    assert compute_shader["num_vectors"] == vectors_per_voxel

    # Test thin shader
    thin_shader = VectorFieldThinShader(wobject)
    assert thin_shader["num_vectors"] == vectors_per_voxel

    # Test regular shader
    reg_shader = VectorFieldShader(wobject)
    assert reg_shader["num_vectors"] == vectors_per_voxel

    # Test arrow shader
    arrow_shader = VectorFieldArrowShader(wobject)
    assert arrow_shader["num_vectors"] == vectors_per_voxel


def test_SphGlyphComputeShader_initialization():
    """Test SphGlyphComputeShader initialization."""
    coefficients = np.random.rand(5, 5, 5, 15)
    n_coeffs = coefficients.shape[-1]  # Exclude the last dimension (vector components)
    wobject = SphGlyph(coefficients, sphere=prim_sphere(name="repulsion100"))
    shader = SphGlyphComputeShader(wobject)

    assert shader["n_coeffs"] == n_coeffs
    assert shader["data_shape"] == (5, 5, 5)
    assert shader["workgroup_size"] == (64, 1, 1)
    assert shader.type == "compute"


def test_SphGlyphComputeShader_get_render_info():
    """Test SphGlyphComputeShader.get_render_info()."""
    coefficients = np.random.rand(5, 5, 5, 15)
    wobject = SphGlyph(coefficients, sphere=prim_sphere(name="repulsion100"))
    shader = SphGlyphComputeShader(wobject)

    render_info = shader.get_render_info(wobject, {})
    assert isinstance(render_info, dict)
    assert "indices" in render_info
    assert render_info["indices"][0] > 0  # Should have at least one workgroup


def test_SphGlyphComputeShader_get_pipeline_info():
    """Test SphGlyphComputeShader.get_pipeline_info()."""
    coefficients = np.random.rand(5, 5, 5, 15)
    wobject = SphGlyph(coefficients, sphere=prim_sphere(name="repulsion100"))
    shader = SphGlyphComputeShader(wobject)

    pipeline_info = shader.get_pipeline_info(wobject, {})
    assert isinstance(pipeline_info, dict)
    assert pipeline_info == {}


def test_SphGlyphComputeShader_get_code():
    """Test SphGlyphComputeShader.get_code()."""
    coefficients = np.random.rand(5, 5, 5, 15)
    wobject = SphGlyph(coefficients, sphere=prim_sphere(name="repulsion100"))
    shader = SphGlyphComputeShader(wobject)
    code = shader.get_code()
    assert isinstance(code, str)
    assert load_wgsl("sph_glyph_compute.wgsl", package_name="fury.wgsl") == code
