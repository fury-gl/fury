import math

import numpy as np

from fury.actor import SphGlyph, VectorField
from fury.actor.curved import (
    Streamlines,
    _create_streamtube_baked,
    _register_streamtube_baking_shaders,
)
from fury.actor.planar import LineProjection
from fury.geometry import line_buffer_separator
from fury.lib import MeshPhongShader, load_wgsl
from fury.material import StreamtubeMaterial
from fury.primitive import prim_sphere
from fury.shader import (
    LineProjectionComputeShader,
    SphGlyphComputeShader,
    StreamlinesShader,
    VectorFieldArrowShader,
    VectorFieldComputeShader,
    VectorFieldShader,
    VectorFieldThinShader,
    _StreamtubeBakingShader,
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


def _make_gpu_streamtube():
    lines = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.5, 0.5]], dtype=np.float32),
        np.array([[0.2, 0.1, 0.0], [0.6, 0.4, 0.5]], dtype=np.float32),
    ]
    colors = np.array([[0.9, 0.1, 0.0], [0.0, 0.5, 0.9]], dtype=np.float32)
    return _create_streamtube_baked(lines, colors=colors, segments=4, end_caps=True)


def test__StreamtubeBakingShader_setup_and_bindings():
    """Streamtube compute shader exposes expected bindings and uniforms."""
    wobject = _make_gpu_streamtube()
    shader = _StreamtubeBakingShader(wobject)

    assert shader.type == "compute"
    assert shader["n_lines"] == wobject.n_lines
    assert shader["max_line_length"] == wobject.max_line_length
    assert shader["tube_sides"] == wobject.tube_sides
    assert math.isclose(shader["tube_radius"], wobject.material.radius)
    assert shader["end_caps"] == 1
    assert shader["color_channels"] == wobject.color_components

    workgroup = min(64, max(int(wobject.n_lines), 1))
    expected_groups = int(math.ceil(wobject.n_lines / workgroup))
    render_info = shader.get_render_info(wobject, {})
    assert render_info["indices"] == (expected_groups, 1, 1)

    bindings = shader.get_bindings_info(wobject, {})
    assert set(bindings.keys()) == {0}
    assert set(bindings[0].keys()) == {0, 1, 2, 3, 4, 5, 6, 7, 8}
    assert bindings[0][0].resource is wobject.line_buffer
    assert bindings[0][5].resource is wobject.geometry.positions
    assert isinstance(shader.get_code(), str)


def test__StreamtubeBakingShader_zero_lines_dispatch():
    """Dispatch collapses to zero groups when no lines are rendered."""
    wobject = _make_gpu_streamtube()
    shader = _StreamtubeBakingShader(wobject)
    wobject.n_lines = 0

    render_info = shader.get_render_info(wobject, {})
    assert render_info["indices"] == (0, 1, 1)


def test__register_streamtube_baking_shaders():
    """Registering GPU streamtube shaders returns compute + phong shader."""
    wobject = _make_gpu_streamtube()
    compute_shader, render_shader = _register_streamtube_baking_shaders(wobject)

    assert isinstance(compute_shader, _StreamtubeBakingShader)
    assert isinstance(render_shader, MeshPhongShader)


def test_streamtube_render_shader_auto_detach_switches_material_and_clears_buffers():
    """Render shader swaps to StreamtubeMaterial and clears compute buffers."""
    wobject = _make_gpu_streamtube()
    compute_shader, render_shader = _register_streamtube_baking_shaders(wobject)

    # Pretend compute finished (render wrapper should swap material once)
    wobject._needs_gpu_update = False
    assert hasattr(wobject, "line_buffer")
    assert hasattr(wobject, "length_buffer")
    assert hasattr(wobject, "color_buffer")
    assert hasattr(wobject, "vertex_offset_buffer")
    assert hasattr(wobject, "triangle_offset_buffer")

    # Trigger render info generation; MeshPhongShader may expect pygfx vars.
    # We only need the auto-detach side-effect; ignore template var KeyErrors.
    _ = compute_shader.get_render_info(wobject, {})
    try:
        _ = render_shader.get_render_info(wobject, {})
    except KeyError:
        pass

    assert isinstance(wobject.material, StreamtubeMaterial)
    for attr in (
        "line_buffer",
        "length_buffer",
        "color_buffer",
        "vertex_offset_buffer",
        "triangle_offset_buffer",
    ):
        assert not hasattr(wobject, attr)


def test__StreamtubeBakingShader_single_dispatch():
    """GPU streamtube compute shader dispatches only once unless flagged."""
    wobject = _make_gpu_streamtube()
    shader = _StreamtubeBakingShader(wobject)

    first = shader.get_render_info(wobject, {})["indices"]
    assert first[0] > 0

    second = shader.get_render_info(wobject, {})["indices"]
    assert second == (0, 1, 1)

    wobject._needs_gpu_update = True
    third = shader.get_render_info(wobject, {})["indices"]
    assert third[0] > 0


def test_streamline_shader_get_code():
    """Test StreamlineShader.get_code()."""
    # Create sample lines data for Streamline constructor
    lines = [np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])]
    lines_positions, lines_colors = line_buffer_separator(lines, color=(1, 0, 0))
    print(lines_positions)
    wobject = Streamlines(lines_positions, colors=lines_colors)
    shader = StreamlinesShader(wobject)
    code = shader.get_code()
    assert isinstance(code, str)
    assert load_wgsl("streamline_render.wgsl", package_name="fury.wgsl") == code


def test_LineProjectionComputeShader_initialization():
    """Test LineProjectionComputeShader initialization."""
    lines = [
        np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
        np.array([[3, 3, 3], [4, 4, 4]]),
    ]
    wobject = LineProjection(lines)
    shader = LineProjectionComputeShader(wobject)

    assert shader["num_lines"] == wobject.num_lines
    assert shader["num_lines"] == 2
    assert shader["workgroup_size"] == 64
    assert shader.type == "compute"


def test_LineProjectionComputeShader_get_render_info():
    """Test LineProjectionComputeShader.get_render_info()."""
    lines = [
        np.array([[0, 0, 0], [1, 1, 1]]),
        np.array([[2, 2, 2], [3, 3, 3]]),
        np.array([[4, 4, 4], [5, 5, 5]]),
    ]
    wobject = LineProjection(lines)
    shader = LineProjectionComputeShader(wobject)

    render_info = shader.get_render_info(wobject, {})
    assert isinstance(render_info, dict)
    assert "indices" in render_info
    assert render_info["indices"][0] > 0  # Should have at least one workgroup

    # Test specific calculation: ceil(num_lines / workgroup_size)
    expected_workgroups = int(np.ceil(wobject.num_lines / shader["workgroup_size"]))
    assert render_info["indices"][0] == expected_workgroups
    assert render_info["indices"] == (expected_workgroups, 1, 1)


def test_LineProjectionComputeShader_get_pipeline_info():
    """Test LineProjectionComputeShader.get_pipeline_info()."""
    lines = [np.array([[0, 0, 0], [1, 1, 1]])]
    wobject = LineProjection(lines)
    shader = LineProjectionComputeShader(wobject)

    pipeline_info = shader.get_pipeline_info(wobject, {})
    assert isinstance(pipeline_info, dict)
    assert pipeline_info == {}


def test_LineProjectionComputeShader_get_code():
    """Test LineProjectionComputeShader.get_code()."""
    lines = [np.array([[0, 0, 0], [1, 1, 1]])]
    wobject = LineProjection(lines)
    shader = LineProjectionComputeShader(wobject)

    code = shader.get_code()
    assert isinstance(code, str)
    assert len(code) > 0
    assert load_wgsl("line_projection_compute.wgsl", package_name="fury.wgsl") == code


def test_LineProjectionComputeShader_with_different_line_configurations():
    """Test LineProjectionComputeShader with different line configurations."""
    # Test with single line
    single_line = [np.array([[0, 0, 0], [1, 1, 1]])]
    wobject_single = LineProjection(single_line)
    shader_single = LineProjectionComputeShader(wobject_single)

    assert shader_single["num_lines"] == 1
    render_info_single = shader_single.get_render_info(wobject_single, {})
    assert render_info_single["indices"][0] == 1  # Should have 1 workgroup for 1 line

    # Test with many lines
    many_lines = [np.array([[i, i, i], [i + 1, i + 1, i + 1]]) for i in range(100)]
    wobject_many = LineProjection(many_lines)
    shader_many = LineProjectionComputeShader(wobject_many)

    assert shader_many["num_lines"] == 100
    render_info_many = shader_many.get_render_info(wobject_many, {})
    expected_workgroups = int(np.ceil(100 / 64))  # 100 lines, 64 workgroup size
    assert render_info_many["indices"][0] == expected_workgroups

    # Test with empty lines (single point lines)
    empty_lines = [np.array([[0, 0, 0]])]
    wobject_empty = LineProjection(empty_lines)
    shader_empty = LineProjectionComputeShader(wobject_empty)

    assert shader_empty["num_lines"] == 1


def test_LineProjectionComputeShader_with_custom_parameters():
    """Test LineProjectionComputeShader with LineProjection custom parameters."""
    lines = [
        np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
        np.array([[3, 3, 3], [4, 4, 4]]),
    ]

    # Test with custom plane
    wobject = LineProjection(
        lines,
        plane=(1, 0, 0, -1),
        colors=[(1, 0, 0), (0, 1, 0)],
        thickness=2.0,
        outline_thickness=0.5,
        opacity=0.8,
    )
    shader = LineProjectionComputeShader(wobject)

    assert shader["num_lines"] == 2
    assert wobject.plane[0] == 1


def test_LineProjectionComputeShader_inheritance():
    """Test LineProjectionComputeShader inheritance and base class functionality."""
    lines = [np.array([[0, 0, 0], [1, 1, 1]])]
    wobject = LineProjection(lines)
    shader = LineProjectionComputeShader(wobject)

    # Should inherit from BaseShader
    assert hasattr(shader, "type")
    assert hasattr(shader, "get_pipeline_info")
    assert hasattr(shader, "get_render_info")
    assert hasattr(shader, "get_bindings")
    assert hasattr(shader, "get_code")

    # Should have compute shader type
    assert shader.type == "compute"
