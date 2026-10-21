import re
import warnings

import numpy as np
import pytest

from fury import actor
from fury.actor import Group
from fury.actor.slicer import (
    _create_chunked_ellipsoid_field,
    _iter_ellipsoid_tiles,
    _max_voxels_per_chunk,
)
from fury.actor.tests._helpers import (
    render_manager,
    render_snapshot,
    snapshot_mask,
)
from fury.actor.utils import (
    get_slices,
    set_group_opacity,
    set_group_visibility,
    show_slices,
)
from fury.lib import gfx_wgpu
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
    """Test opacity validation raises errors for out-of-bounds values."""
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
    vf_group = actor.vector_field(field, actor_type="arrow", opacity=0.5, thickness=2.0)
    assert vf_group.name == "VectorField"
    assert len(vf_group.children) == 1
    vf = vf_group.children[0]
    assert isinstance(vf.material, VectorFieldArrowMaterial)
    assert vf.material.opacity == 0.5
    assert vf.material.thickness == 2.0

    # Test vector_field_slicer
    slicer_group = actor.vector_field_slicer(
        field,
        actor_type="line",
        cross_section=[2, 2, 2],
        visibility=(True, False, True),
    )
    assert slicer_group.name == "VectorFieldSlicer"
    assert len(slicer_group.children) == 1
    vfs = slicer_group.children[0]
    assert isinstance(vfs.material, VectorFieldLineMaterial)
    assert np.all(vfs.cross_section == np.array([2, 2, 2]))
    assert np.all(vfs.visibility == np.asarray((True, False, True)))


@pytest.mark.parametrize(
    ("max_buffer_size", "max_workgroups", "expected"),
    [
        (24, 65535, 1),
        (24 * 100, 1, 64),
        (2 * 3 * 4 * 100, 2, min(100, 2 * 64)),
    ],
)
def test_max_voxels_per_chunk(max_buffer_size, max_workgroups, expected):
    """Storage-buffer and dispatch limits both cap chunk voxel count."""
    got = _max_voxels_per_chunk(
        max_buffer_size=max_buffer_size,
        max_workgroups=max_workgroups,
    )
    assert got == expected


def test_vector_field_cross_section_world_bounds_with_chunk_offset():
    """
    Cross section clamps in world space when the actor has a local X
    offset.
    """
    field = np.random.rand(5, 5, 5, 3)
    vf = actor.VectorField(field)
    vf.local.position = [10.0, 0.0, 0.0]
    vf.cross_section = [20, 2, 2]
    np.testing.assert_array_equal(vf.cross_section, np.array([4, 2, 2]))


def test_vector_field_chunks_when_exceeding_device_limits():
    """
    Large fields are split into several actors along X with contiguous
    offsets.
    """
    device = gfx_wgpu.get_shared().device
    max_voxels = _max_voxels_per_chunk(
        max_buffer_size=device.limits.get(
            "max-storage-buffer-binding-size", 256 * 1024 * 1024
        ),
        max_workgroups=device.limits.get("max-compute-workgroups-per-dimension", 65535),
    )
    y, z = 32, 32
    x = int(max_voxels // (y * z)) + 5
    shape = (x, y, z, 3)
    total = int(np.prod(shape[:3]))
    field = np.random.default_rng(0).random(shape, dtype=np.float32)
    scales = np.random.default_rng(1).random(shape[:3], dtype=np.float32)

    group = actor.vector_field(field, scales=scales)
    assert group.name == "VectorField"
    if total > max_voxels:
        assert len(group.children) >= 2
        assert sum(ch.field_shape[0] for ch in group.children) == x
        x_cursor = 0
        for ch in group.children:
            assert ch.local.position[0] == x_cursor
            x_cursor += ch.field_shape[0]


def test_vector_field_slicer_chunks_when_exceeding_device_limits():
    """vector_field_slicer uses the same chunking strategy as vector_field."""
    device = gfx_wgpu.get_shared().device
    max_voxels = _max_voxels_per_chunk(
        max_buffer_size=device.limits.get(
            "max-storage-buffer-binding-size", 256 * 1024 * 1024
        ),
        max_workgroups=device.limits.get("max-compute-workgroups-per-dimension", 65535),
    )
    y, z = 24, 24
    x = int(max_voxels // (y * z)) + 3
    shape = (x, y, z, 3)
    total = int(np.prod(shape[:3]))
    field = np.random.default_rng(2).random(shape, dtype=np.float32)
    cs = np.array([min(5, x - 1), y // 2, z // 2], dtype=np.int32)

    group = actor.vector_field_slicer(field, cross_section=cs)
    assert group.name == "VectorFieldSlicer"
    if total > max_voxels:
        assert len(group.children) >= 2
        assert sum(ch.field_shape[0] for ch in group.children) == x
    else:
        assert len(group.children) == 1
        np.testing.assert_array_equal(group.children[0].cross_section, cs)


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


@pytest.mark.parametrize("latitude_segments, longitude_segments", [(4, 8), (8, 4)])
def test_sph_glyph_latitude_longitude_resolution(latitude_segments, longitude_segments):
    """
    Honor custom glyph latitude and longitude sampling with outward pole fans.

    Parameters
    ----------
    latitude_segments : int
        Number of segments between the poles.
    longitude_segments : int
        Number of segments around the equator.
    """
    coeffs = np.zeros((1, 1, 1, 9), dtype=np.float32)
    coeffs[..., 0] = 1.0
    glyph = actor.sph_glyph(coeffs, sphere=(latitude_segments, longitude_segments))

    vertices = glyph.sphere
    grid = vertices.reshape(latitude_segments + 1, longitude_segments + 1, 3)
    latitude = np.linspace(-np.pi / 2, np.pi / 2, latitude_segments + 1)
    longitude = np.linspace(-np.pi, np.pi, longitude_segments + 1)
    np.testing.assert_allclose(
        grid[:, :, 1],
        np.broadcast_to(np.sin(latitude)[:, None], grid.shape[:2]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        grid[latitude_segments // 2, :, 0], np.sin(longitude), atol=1e-6
    )
    np.testing.assert_allclose(
        grid[latitude_segments // 2, :, 2], np.cos(longitude), atol=1e-6
    )
    np.testing.assert_allclose(grid[:, 0], grid[:, -1], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), 1, atol=1e-6)

    faces = glyph.indices.reshape(-1, 3)
    expected_faces = 2 * longitude_segments * (latitude_segments - 1)
    assert faces.shape == (expected_faces, 3)
    assert glyph.faces_per_glyph == expected_faces
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    assert np.all(np.linalg.norm(normals, axis=1) > 1e-7)
    assert np.all(np.sum(normals * triangles.mean(axis=1), axis=1) > 0)


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


def _ellipsoid_field_pixel(camera, center):
    """Project a world center into the fixed 256-pixel test viewport."""
    clip = camera.camera_matrix @ np.append(center, 1)
    ndc = clip[:2] / clip[3]
    return tuple(np.floor((ndc * (1, -1) + 1) * 128).astype(int))


def _ellipsoid_field_pick(show_m, camera, center, voxel):
    """Assert real surface picking and the tile-to-global voxel mapping."""
    point = _ellipsoid_field_pixel(camera, center)
    pick = show_m.renderer.get_pick_info(point)
    child = pick["world_object"]
    assert child is not None
    assert isinstance(child.parent, Group)
    assert child in child.parent.children
    assert pick["voxel_index"] == voxel
    local_voxel = np.asarray(voxel) - child.voxel_origin
    assert pick["glyph_index"] == np.ravel_multi_index(
        tuple(local_voxel), child.field_shape
    )
    return point


@pytest.mark.parametrize("default_colors", [False, True])
def test_ellipsoid_slicer_chunk_boundaries(default_colors):
    """Scaled XYZ tiles retain unscaled strided views and seamless surface picks."""
    shape = (3, 5, 7)
    lengths = np.full((6, 10, 14, 3), 0.2, dtype=np.float32)[::2, ::2, ::2]
    lengths[..., 0] = 0.25
    orientations = np.broadcast_to(
        np.eye(3, dtype=np.float32), (6, 10, 14, 3, 3)
    ).copy()[::2, ::2, ::2]
    orientations[1, ..., :2, :2] = [[0, -1], [1, 0]]
    orientations[1, :, 4:] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    mask = np.ones((6, 10, 14), dtype=bool)[::2, ::2, ::2]
    mask[1, 2, 3] = False
    mask[2, 4, 6] = False
    colors = np.zeros((6, 10, 14, 3), dtype=np.uint8)[::2, ::2, ::2]
    colors[..., 0] = 50
    colors[..., 1] = np.arange(5, dtype=np.uint8)[None, :, None] * 40
    colors[..., 2] = np.arange(7, dtype=np.uint8)[None, None, :] * 40
    # This complete tile must retain the full field's 255-based convention.
    colors[1, 0] = (1, 0, 0)
    lengths[~mask] = np.nan
    orientations[~mask] = np.nan
    if default_colors:
        colors = None
    options = {
        "cross_section": (1, 2, 3),
        "visibility": (True, False, False),
        "opacity": 1,
        "enable_picking": True,
    }
    tiles = list(_iter_ellipsoid_tiles(shape, 10))
    assert [tuple((axis.start, axis.stop) for axis in tile) for tile in tiles] == [
        ((x, x + 1), (y, y + 1), (0, 7)) for x in range(3) for y in range(5)
    ]
    coverage = np.zeros(shape, dtype=np.uint8)
    for tile in tiles:
        coverage[tile] += 1
        assert np.prod(coverage[tile].shape) <= 10
    np.testing.assert_array_equal(coverage, 1)

    images = []
    picked_voxels = []
    for capacity in (105, 10, 4):
        group = _create_chunked_ellipsoid_field(
            lengths,
            orientation_matrices=orientations,
            mask=mask,
            colors=colors,
            actor_params=options,
            max_glyphs=capacity,
            scale=0.65,
        )
        assert isinstance(group, Group)
        assert group.name == "EllipsoidFieldSlicer"
        expected_tiles = list(_iter_ellipsoid_tiles(shape, capacity))
        assert len(group.children) == len(expected_tiles)
        for child, tile in zip(group.children, expected_tiles, strict=True):
            tile_shape = lengths[tile].shape[:3]
            assert child.field_shape == tile_shape
            assert tuple(child.voxel_origin) == tuple(axis.start for axis in tile)
            assert child.glyph_count == np.prod(tile_shape) <= capacity
            assert child.geometry.positions.data.nbytes <= capacity * 12
            assert child.geometry.colors.data.nbytes <= capacity * 16
            assert child.geometry.ellipsoid_axes.data.nbytes <= capacity * 48
            np.testing.assert_allclose(child.local.matrix, np.eye(4))
            np.testing.assert_array_equal(child.bounds, [[0, 0, 0], [2, 4, 6]])
            for attribute, original in (
                ("lengths", lengths),
                ("orientation_matrices", orientations),
                ("mask", mask),
                ("colors", colors),
            ):
                if original is None:
                    assert getattr(child, attribute) is None
                    continue
                retained = getattr(child, attribute)
                assert np.shares_memory(retained, original)
                np.testing.assert_array_equal(retained, original[tile])
        with render_manager(group, target=(1, 2, 3), position=(11, 2, 3)) as show_m:
            camera = show_m.screens[0].camera
            image = render_snapshot(show_m)
            images.append(image)
            voxels = []
            # Y boundaries at capacity 10 and Z boundaries at capacity 4.
            for voxel in ((1, 0, 3), (1, 1, 3), (1, 1, 4), (1, 4, 6)):
                point = _ellipsoid_field_pick(show_m, camera, voxel, voxel)
                voxels.append(show_m.renderer.get_pick_info(point)["voxel_index"])
            assert (
                show_m.renderer.get_pick_info(
                    _ellipsoid_field_pixel(camera, (1, 2, 3))
                )["world_object"]
                is None
            )
            first_color = _ellipsoid_field_pixel(camera, (1, 0, 3))
            last_color = _ellipsoid_field_pixel(camera, (1, 4, 6))
            if default_colors:
                first_rgb = image[first_color[1], first_color[0], :3]
                last_rgb = image[last_color[1], last_color[0], :3]
                assert first_rgb[1] > 2 * max(int(first_rgb[0]), int(first_rgb[2]))
                assert last_rgb[2] > 2 * max(int(last_rgb[0]), int(last_rgb[1]))
            else:
                assert image[first_color[1], first_color[0], :3].max() <= 3
                assert image[last_color[1], last_color[0], :3].max() > 20
            picked_voxels.append(voxels)
    for image, voxels in zip(images[1:], picked_voxels[1:], strict=True):
        np.testing.assert_allclose(image, images[0], atol=1)
        assert voxels == picked_voxels[0]


@pytest.mark.parametrize(
    "case", ["translated_scale", "inclusive_boundary", "zero_diagonal", "shear"]
)
def test_ellipsoid_slice_distance_selection(case):
    """World-plane distance, inclusive snapping and diagonal tolerance match vectors."""
    lengths = np.full((3, 2, 1, 3), 0.16, dtype=np.float32)
    mask = np.zeros((3, 2, 1), dtype=bool)
    active = ((0, 0, 0), (1, 1, 0), (2, 0, 0))
    for voxel in active:
        mask[voxel] = True
    vectors = np.zeros_like(lengths)
    vectors[mask] = (0, 0.16, 0)
    affine = np.eye(4)
    affine[0, 0] = 2
    affine[0, 3] = 0.25
    selections = [
        ((1, 0, 0), (True, False, False), [((0, 0, 0), (1, 0, 0))]),
        ((2, 0, 0), (True, False, False), [((1, 1, 0), (2, 1, 0))]),
    ]
    if case == "inclusive_boundary":
        affine[0, 3] = 0
        selections = [
            (
                (1, 0, 0),
                (True, False, False),
                [((0, 0, 0), (1, 0, 0)), ((1, 1, 0), (1, 1, 0))],
            )
        ]
    elif case == "zero_diagonal":
        affine[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        selections = [
            ((0, 1, 0), (True, False, False), []),
            ((0, 1, 0), (False, True, False), [((1, 1, 0), (-0.75, 1, 0))]),
        ]
    elif case == "shear":
        affine[0, 1] = 0.5
    corners = np.array([[x, y, 0, 1] for x in (0, 2) for y in (0, 1)], dtype=float)
    transformed = (affine @ corners.T).T[:, :3]
    bounds = np.array([transformed.min(axis=0), transformed.max(axis=0)])
    ellipsoids = actor.ellipsoid_slicer(lengths, mask=mask, colors=(1, 1, 1))
    vector_group = actor.vector_field_slicer(
        vectors, actor_type="thin_line", colors=(1, 1, 1)
    )
    for group in (ellipsoids, vector_group):
        group.bounds = bounds
        for child in group.children:
            child.local.state_basis = "matrix"
            child.local.matrix = affine
            child.bounds = bounds
        with render_manager(group, target=(2, 0.5, 0), position=(2, 0.5, 10)) as show_m:
            camera = show_m.screens[0].camera
            for section, visibility, expected in selections:
                for child in group.children:
                    child.visibility = visibility
                show_slices(group, section)
                image = render_snapshot(show_m)
                foreground = snapshot_mask(image)
                allowed = np.zeros((256, 256), dtype=bool)
                for voxel, center in expected:
                    x, y = _ellipsoid_field_pixel(camera, center)
                    region = np.s_[y - 14 : y + 15, x - 14 : x + 15]
                    assert foreground[region].any()
                    allowed[region] = True
                    if group is ellipsoids:
                        _ellipsoid_field_pick(show_m, camera, center, voxel)
                assert not foreground[~allowed].any()
                if not expected:
                    assert not foreground.any()
                    for voxel in active:
                        center = (affine @ np.append(voxel, 1))[:3]
                        assert (
                            show_m.renderer.get_pick_info(
                                _ellipsoid_field_pixel(camera, center)
                            )["world_object"]
                            is None
                        )


def test_ellipsoid_slicer_controls_and_picking():
    """Slice uniforms, plane unions and ordinary opacity/visibility control surfaces."""
    lengths = np.full((3, 3, 1, 3), 0.28, dtype=np.float32)
    mask = np.ones((3, 3, 1), dtype=bool)
    mask[0, 0, 0] = False
    lengths[0, 0, 0] = np.nan
    lengths[2, 2, 0, 2] = 0
    group = _create_chunked_ellipsoid_field(
        lengths,
        orientation_matrices=None,
        mask=mask,
        colors=(1, 0.3, 0.1),
        actor_params={
            "cross_section": (0, 1, 0),
            "visibility": (True, False, False),
            "opacity": 1,
            "enable_picking": True,
        },
        max_glyphs=3,
    )
    buffers = [
        (
            child.geometry.positions,
            child.geometry.colors,
            child.geometry.ellipsoid_axes,
        )
        for child in group.children
    ]
    original_data = [[buffer.data.copy() for buffer in row] for row in buffers]
    with render_manager(
        group, width=6, target=(1, 1, 0), position=(1, 1, 10)
    ) as show_m:
        camera = show_m.screens[0].camera
        initial = render_snapshot(show_m)
        _ellipsoid_field_pick(show_m, camera, (0, 1, 0), (0, 1, 0))
        assert (
            show_m.renderer.get_pick_info(_ellipsoid_field_pixel(camera, (1, 1, 0)))[
                "world_object"
            ]
            is None
        )
        show_slices(group, (1, 1, 0))
        moved = render_snapshot(show_m)
        assert not np.array_equal(initial, moved)
        for y in range(3):
            _ellipsoid_field_pick(show_m, camera, (1, y, 0), (1, y, 0))
        assert (
            show_m.renderer.get_pick_info(_ellipsoid_field_pixel(camera, (0, 1, 0)))[
                "world_object"
            ]
            is None
        )

        expected_by_visibility = [
            ((True, False, False), {(1, 0, 0), (1, 1, 0), (1, 2, 0)}),
            ((False, True, False), {(0, 1, 0), (1, 1, 0), (2, 1, 0)}),
            (
                (False, False, True),
                {
                    (x, y, 0)
                    for x in range(3)
                    for y in range(3)
                    if (x, y) not in ((0, 0), (2, 2))
                },
            ),
            ((False, False, False), set()),
        ]
        for visibility, expected in expected_by_visibility:
            for child in group.children:
                child.visibility = visibility
            image = render_snapshot(show_m)
            for x in range(3):
                for y in range(3):
                    voxel = (x, y, 0)
                    point = _ellipsoid_field_pixel(camera, voxel)
                    if voxel in expected:
                        _ellipsoid_field_pick(show_m, camera, voxel, voxel)
                        assert image[point[1], point[0], :3].max() > 5
                    else:
                        assert (
                            show_m.renderer.get_pick_info(point)["world_object"] is None
                        )
                        assert image[point[1], point[0], :3].max() == 0

        point = _ellipsoid_field_pixel(camera, (1, 1, 0))
        set_group_opacity(group, 0.35)
        intersection_colors = []
        for visibility in (
            (True, False, False),
            (False, True, False),
            (True, True, False),
        ):
            for child in group.children:
                child.visibility = visibility
            image = render_snapshot(show_m)
            intersection_colors.append(image[point[1], point[0], :3])
        np.testing.assert_allclose(
            intersection_colors[1:], [intersection_colors[0]] * 2, atol=1
        )
        assert 0 < intersection_colors[0][0] < moved[point[1], point[0], 0]
        selected_child = group.children[1]
        selected_child.opacity = 0.15
        faded = render_snapshot(show_m)
        assert 0 < faded[point[1], point[0], 0] < intersection_colors[0][0]
        set_group_opacity(group, 1)
        group.visible = False
        assert not render_snapshot(show_m)[..., :3].any()
        assert show_m.renderer.get_pick_info(point)["world_object"] is None
        group.visible = True
        set_group_visibility(group, False)
        assert not render_snapshot(show_m)[..., :3].any()
        assert show_m.renderer.get_pick_info(point)["world_object"] is None
        set_group_visibility(group, True)
        set_group_visibility(group, (True, False, True))
        for child in group.children:
            child.visibility = (True, False, False)
        show_slices(group, (1, 1, 0))
        assert not selected_child.visible
        assert not render_snapshot(show_m)[..., :3].any()
        assert show_m.renderer.get_pick_info(point)["world_object"] is None
        show_slices(group, (0, 1, 0))
        render_snapshot(show_m)
        _ellipsoid_field_pick(show_m, camera, (0, 1, 0), (0, 1, 0))
        assert not selected_child.visible
        for child, row, data in zip(
            group.children, buffers, original_data, strict=True
        ):
            assert child.geometry.positions is row[0]
            assert child.geometry.colors is row[1]
            assert child.geometry.ellipsoid_axes is row[2]
            for buffer, original in zip(row, data, strict=True):
                np.testing.assert_array_equal(buffer.data, original)


@pytest.mark.parametrize(
    "options",
    [
        {"orientation_matrices": np.ones((2, 3, 3))},
        {"mask": np.ones((2, 2), dtype=bool)},
        {"colors": np.ones((2, 2, 2, 2))},
        {"cross_section": (0, 0)},
        {"visibility": (True, False)},
        {"opacity": -0.1},
    ],
)
def test_ellipsoid_slicer_invalid_options(options):
    """Invalid field options fail rather than altering requested behavior."""
    with pytest.raises(ValueError):
        actor.ellipsoid_slicer(np.ones((2, 2, 2, 3)), **options)


@pytest.mark.parametrize("shape", [(2, 2, 3), (2, 2, 2, 2), (0, 2, 2, 3)])
def test_ellipsoid_slicer_invalid_shape(shape):
    """A field requires positive XYZ dimensions and three semi-axes."""
    with pytest.raises(ValueError):
        actor.ellipsoid_slicer(np.ones(shape))


@pytest.mark.parametrize("capacity", [0, -1, 1.5, True])
def test_ellipsoid_slicer_invalid_private_capacity(capacity):
    """The private override accepts only positive integer capacities."""
    with pytest.raises(ValueError):
        _create_chunked_ellipsoid_field(
            np.ones((1, 1, 1, 3)),
            orientation_matrices=None,
            mask=None,
            colors=(1, 1, 1),
            actor_params={},
            max_glyphs=capacity,
        )


def test_ellipsoid_slicer_defaults_uniform_orientation_and_disabled_picking():
    """Uniform orientations retain ownership and disabled picks leave visible glyphs."""
    lengths = np.full((2, 3, 1, 3), 0.2, dtype=np.float32)
    orientation = np.eye(3, dtype=np.float32)
    group = actor.ellipsoid_slicer(
        lengths, orientation_matrices=orientation, enable_picking=False
    )
    np.testing.assert_array_equal(get_slices(group), [1, 1, 0])
    np.testing.assert_array_equal(group.bounds, [[0, 0, 0], [1, 2, 0]])
    for child in group.children:
        assert np.shares_memory(child.orientation_matrices, orientation)
    show_slices(group, (-10, 20, 10))
    np.testing.assert_array_equal(get_slices(group), [0, 2, 0])
    with render_manager(group, target=(1, 1, 0), position=(1, 1, 10)) as show_m:
        camera = show_m.screens[0].camera
        image = render_snapshot(show_m)
        point = _ellipsoid_field_pixel(camera, (1, 1, 0))
        assert image[point[1], point[0], :3].max() > 5
        assert show_m.renderer.get_pick_info(point)["world_object"] is None


@pytest.mark.parametrize("scale", [0.0, 1.0])
@pytest.mark.parametrize("value", [-0.1, np.nan, np.inf])
def test_ellipsoid_slicer_invalid_active_lengths(value, scale):
    """Invalid original lengths cannot become valid through zero scaling."""
    lengths = np.full((2, 2, 1, 3), 0.2)
    lengths[1, 1, 0, 0] = value
    with (
        warnings.catch_warnings(),
        np.errstate(invalid="raise", divide="raise", over="raise"),
    ):
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError):
            actor.ellipsoid_slicer(lengths, scale=scale)


@pytest.mark.parametrize("value", [np.nan, np.inf])
@pytest.mark.parametrize("scale", [0.0, 1.0])
def test_ellipsoid_slicer_invalid_active_orientation(value, scale):
    """Invalid active directions fail before color arithmetic emits warnings."""
    lengths = np.full((2, 2, 1, 3), 0.2)
    orientations = np.broadcast_to(np.eye(3), (2, 2, 1, 3, 3)).copy()
    orientations[1, 1, 0, 0, 0] = value
    with (
        warnings.catch_warnings(),
        np.errstate(invalid="raise", divide="raise", over="raise"),
    ):
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError):
            actor.ellipsoid_slicer(
                lengths, orientation_matrices=orientations, scale=scale
            )


def test_ellipsoid_slicer_rgba_field_colors():
    """Spatial RGBA colors and alpha are observable on real picked surfaces."""
    lengths = np.full((2, 1, 1, 3), 0.3, dtype=np.float32)
    colors = np.array([[[[255, 0, 0, 255]]], [[[0, 0, 255, 128]]]], dtype=np.uint8)
    orientation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    group = actor.ellipsoid_slicer(
        lengths, orientation_matrices=orientation, colors=colors
    )
    with render_manager(
        group, target=(0.5, 0, 0), position=(0.5, 0, 10), width=4
    ) as show_m:
        camera = show_m.screens[0].camera
        image = render_snapshot(show_m)
        pixels = []
        for x in range(2):
            point = _ellipsoid_field_pick(show_m, camera, (x, 0, 0), (x, 0, 0))
            pixels.append(image[point[1], point[0], :3])
        assert pixels[0][0] > 2 * max(int(pixels[0][1]), int(pixels[0][2]))
        assert pixels[1][2] > 2 * max(int(pixels[1][0]), int(pixels[1][1]))
        assert 0 < pixels[1][2] < pixels[0][0]


@pytest.mark.parametrize("scale", [0.0, np.float64(0.5), np.array(1.5)])
def test_ellipsoid_slicer_scalar_scale(scale):
    """Scaling changes glyph bounds and silhouettes, never centers or CPU views."""
    lengths = np.broadcast_to(
        np.array((0.24, 0.12, 0.08), dtype=np.float32), (2, 1, 1, 3)
    )
    original_lengths = lengths.copy()
    group = actor.ellipsoid_slicer(lengths, scale=scale)
    center_bounds = np.array([[0, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(group.bounds, center_bounds)
    np.testing.assert_array_equal(get_slices(group), [1, 0, 0])
    assert len(group.children) == 1
    child = group.children[0]
    assert np.shares_memory(child.lengths, lengths)
    np.testing.assert_array_equal(child.lengths, original_lengths)
    np.testing.assert_array_equal(child.bounds, center_bounds)
    # Match the field input precision before comparing rendered pixels.
    radii = original_lengths[0, 0, 0] * float(scale)
    np.testing.assert_allclose(
        child.get_bounding_box(),
        [center_bounds[0] - radii, center_bounds[1] + radii],
        atol=1e-7,
    )

    with render_manager(
        group, target=(0.5, 0, 0), position=(0.5, 0, 10), width=4
    ) as show_m:
        camera = show_m.screens[0].camera
        image = render_snapshot(show_m)
        foreground = snapshot_mask(image)
        if scale == 0:
            assert not foreground.any()
        for x in range(2):
            center = (x, 0, 0)
            point = _ellipsoid_field_pixel(camera, center)
            if scale == 0:
                assert show_m.renderer.get_pick_info(point)["world_object"] is None
                continue
            _ellipsoid_field_pick(show_m, camera, center, center)
            region = foreground[:, x * 128 : (x + 1) * 128]
            rows, columns = np.nonzero(region)
            silhouette_center = (
                (columns.min() + columns.max()) / 2 + x * 128,
                (rows.min() + rows.max()) / 2,
            )
            np.testing.assert_allclose(silhouette_center, point, atol=1)
            rgb = image[point[1], point[0], :3]
            assert rgb[0] > 2 * max(int(rgb[1]), int(rgb[2]))
    reference = actor.ellipsoid(
        [[0, 0, 0], [1, 0, 0]], lengths=radii, colors=(1, 0, 0), impostor=True
    )
    with render_manager(
        reference, target=(0.5, 0, 0), position=(0.5, 0, 10), width=4
    ) as show_m:
        np.testing.assert_allclose(render_snapshot(show_m), image, atol=1)
    np.testing.assert_array_equal(lengths, original_lengths)


@pytest.mark.parametrize(
    "magnitude,dtype,scale", [(1e38, np.float32, 10), (1e300, np.float64, 1e100)]
)
def test_ellipsoid_slicer_scaled_length_range(magnitude, dtype, scale):
    """
    Reject oversized products before scaling arithmetic.

    Parameters
    ----------
    magnitude : float
        Finite original semi-axis length.
    dtype : type
        Input floating-point dtype.
    scale : float
        Multiplier making the product unrepresentable.

    Returns
    -------
    None
        Assertions check the representability error without arithmetic overflow.
    """
    lengths = np.full((1, 1, 1, 3), magnitude, dtype=dtype)
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(ValueError, match="fit in float32"):
            actor.ellipsoid_slicer(lengths, scale=scale)


@pytest.mark.parametrize(
    "magnitude,dtype,scale", [(1e-30, np.float32, 1e40), (1e300, np.float64, 1e-300)]
)
def test_ellipsoid_slicer_extreme_scale_bounds(magnitude, dtype, scale):
    """
    Preserve valid extreme products and the original CPU view.

    Parameters
    ----------
    magnitude : float
        Original semi-axis length.
    dtype : type
        Input floating-point dtype.
    scale : float
        Multiplier producing finite float32-representable bounds.

    Returns
    -------
    None
        Assertions check geometric bounds and input ownership.
    """
    lengths = np.full((1, 1, 1, 3), magnitude, dtype=dtype)
    original = lengths.copy()
    radii = original[0, 0, 0].astype(np.float64) * scale
    group = actor.ellipsoid_slicer(lengths, scale=scale)
    child = group.children[0]
    bounds = child.get_bounding_box()
    assert np.isfinite(bounds).all()
    np.testing.assert_allclose(bounds, [-radii, radii], rtol=1e-6)
    np.testing.assert_array_equal(lengths, original)
    assert np.shares_memory(child.lengths, lengths)


@pytest.mark.parametrize("background", [np.nan, np.inf, 1e300])
def test_ellipsoid_slicer_scaled_masked_background(background):
    """
    Exclude masked invalid lengths from scaling and rendered surfaces.

    Parameters
    ----------
    background : float
        Invalid or oversized masked semi-axis length.

    Returns
    -------
    None
        Assertions check active radii, picking, and all-masked omission.
    """
    lengths = np.array([[0.2, 0.1, 0.1], [background] * 3])[:, None, None, :]
    mask = np.array([True, False])[:, None, None]
    group = actor.ellipsoid_slicer(
        lengths,
        mask=mask,
        scale=2,
        colors=(0, 1, 0),
        visibility=(False, False, True),
    )
    with render_manager(group, width=4) as show_m:
        camera = show_m.screens[0].camera
        image = render_snapshot(show_m)
        px, py = _ellipsoid_field_pick(show_m, camera, (0, 0, 0), (0, 0, 0))
        assert image[py, px, 1] > 50
        point = _ellipsoid_field_pixel(camera, (1, 0, 0))
        assert not np.any(image[point[1], point[0], :3])
        assert show_m.renderer.get_pick_info(point)["world_object"] is None
    reference = actor.ellipsoid(
        [[0, 0, 0]], lengths=(0.4, 0.2, 0.2), colors=(0, 1, 0), impostor=True
    )
    with render_manager(reference, width=4) as show_m:
        np.testing.assert_allclose(render_snapshot(show_m), image, atol=1)
    group = actor.ellipsoid_slicer(lengths, mask=np.zeros_like(mask), scale=1e300)
    with render_manager(group, width=4) as show_m:
        assert not snapshot_mask(render_snapshot(show_m)).any()
        for x in (0, 1):
            point = _ellipsoid_field_pixel(show_m.screens[0].camera, (x, 0, 0))
            assert show_m.renderer.get_pick_info(point)["world_object"] is None


@pytest.mark.parametrize("scale", [-0.1, np.nan, np.inf, -np.inf])
def test_ellipsoid_slicer_invalid_scalar_scale(scale):
    """Even an empty-volume field rejects negative and nonfinite scale."""
    with pytest.raises(ValueError):
        actor.ellipsoid_slicer(np.zeros((1, 1, 1, 3)), scale=scale)
    with pytest.raises(ValueError):
        _create_chunked_ellipsoid_field(
            np.zeros((1, 1, 1, 3)),
            orientation_matrices=None,
            mask=None,
            colors=None,
            scale=scale,
            actor_params={},
            max_glyphs=1,
        )


@pytest.mark.parametrize("orientation_kind", ["identity", "uniform", "field"])
def test_ellipsoid_slicer_principal_direction_colors(orientation_kind):
    """Unsorted and tied axes use normalized absolute input-frame columns."""
    lengths = np.array(
        [
            (0.32, 0.16, 0.12),
            (0.12, 0.32, 0.16),
            (0.12, 0.16, 0.32),
            (0.32, 0.32, 0.12),
        ],
        dtype=np.float32,
    ).reshape(4, 1, 1, 3)
    frame = np.column_stack(
        (
            np.array((1, 2, 3)) / np.sqrt(14),
            np.array((2, -1, 0)) / np.sqrt(5),
            np.array((3, 6, -5)) / np.sqrt(70),
        )
    )
    if orientation_kind == "identity":
        orientations = None
        expected_colors = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0)])
        flipped_orientations = np.diag((-1.0, 1.0, 1.0))
    elif orientation_kind == "uniform":
        orientations = frame
        expected_colors = np.array(
            [
                np.array((1, 2, 3)) / np.sqrt(14),
                np.array((2, 1, 0)) / np.sqrt(5),
                np.array((3, 6, 5)) / np.sqrt(70),
                np.array((1, 2, 3)) / np.sqrt(14),
            ]
        )
        flipped_orientations = frame * (-1, 1, 1)
    else:
        orientations = np.broadcast_to(np.eye(3), (4, 1, 1, 3, 3)).copy()
        orientations[0, 0, 0] = frame
        orientations[2, 0, 0] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        orientations[3, 0, 0] = np.array(
            [[1, -1, 0], [1, 1, 0], [0, 0, np.sqrt(2)]]
        ) / np.sqrt(2)
        expected_colors = np.array(
            [
                np.array((1, 2, 3)) / np.sqrt(14),
                (0, 1, 0),
                (1, 0, 0),
                np.array((1, 1, 0)) / np.sqrt(2),
            ]
        )
        flipped_orientations = orientations * (-1, 1, 1)
    original_lengths = lengths.copy()
    original_orientations = None if orientations is None else orientations.copy()
    groups = (
        actor.ellipsoid_slicer(lengths, orientation_matrices=orientations),
        actor.ellipsoid_slicer(
            lengths,
            orientation_matrices=orientations,
            colors=expected_colors.reshape(4, 1, 1, 3),
        ),
        actor.ellipsoid_slicer(lengths, orientation_matrices=flipped_orientations),
    )
    images = []
    for group in groups:
        with render_manager(
            group, target=(1.5, 0, 0), position=(1.5, 0, 10), width=5
        ) as show_m:
            camera = show_m.screens[0].camera
            image = render_snapshot(show_m)
            images.append(image)
            for x in range(4):
                voxel = (x, 0, 0)
                point = _ellipsoid_field_pick(show_m, camera, voxel, voxel)
                assert image[point[1], point[0], :3].max() > 20
    np.testing.assert_allclose(images[0], images[1], atol=1)
    np.testing.assert_allclose(images[0], images[2], atol=1)
    np.testing.assert_array_equal(lengths, original_lengths)
    if orientations is not None:
        np.testing.assert_array_equal(orientations, original_orientations)


def test_ellipsoid_slicer_direction_colors_under_world_transform():
    """A fully 3D world transform changes shape, not model-frame directional RGB."""
    lengths = np.array([[[[0.4, 0.25, 0.15]]]])
    orientation = np.column_stack(
        (
            np.array((1, 2, 3)) / np.sqrt(14),
            np.array((2, -1, 0)) / np.sqrt(5),
            np.array((3, 6, -5)) / np.sqrt(70),
        )
    )
    transform = np.array(
        [
            [0.8, -0.6, 0.3, 0],
            [0.5, 1.1, -0.4, 0],
            [-0.7, 0.2, 1.2, 0],
            [0, 0, 0, 1],
        ]
    )
    images = []
    for colors in (None, np.array((1, 2, 3)) / np.sqrt(14)):
        group = actor.ellipsoid_slicer(
            lengths, orientation_matrices=orientation, colors=colors
        )
        for child in group.children:
            child.local.state_basis = "matrix"
            child.local.matrix = transform
        with render_manager(group, width=3) as show_m:
            image = render_snapshot(show_m)
            camera = show_m.screens[0].camera
            point = _ellipsoid_field_pick(show_m, camera, (0, 0, 0), (0, 0, 0))
            assert image[point[1], point[0], :3].max() > 20
            images.append(image)
    np.testing.assert_allclose(images[0], images[1], atol=1)


def test_ellipsoid_slicer_direction_colors_original_float64_lengths():
    """Scaling and float32 packing cannot turn an original longest axis into a tie."""
    lengths = np.array([[[[0.3, 0.3 + 1e-9, 0.15]]]], dtype=np.float64)
    assert lengths[0, 0, 0, 1] > lengths[0, 0, 0, 0]
    assert np.float32(lengths[0, 0, 0, 0]) == np.float32(lengths[0, 0, 0, 1])
    images = []
    for colors in (None, (0, 1, 0)):
        group = actor.ellipsoid_slicer(lengths, scale=1.7, colors=colors)
        with render_manager(group, width=3) as show_m:
            image = render_snapshot(show_m)
            camera = show_m.screens[0].camera
            point = _ellipsoid_field_pick(show_m, camera, (0, 0, 0), (0, 0, 0))
            rgb = image[point[1], point[0], :3]
            assert rgb[1] > 20
            assert rgb[1] > 2 * max(int(rgb[0]), int(rgb[2]))
            images.append(image)
    np.testing.assert_allclose(images[0], images[1], atol=1)


def test_ellipsoid_slicer_uniform_color_precedence():
    """Explicit hex, RGB and RGBA override principal-axis colors and retain alpha."""
    lengths = np.array([[[[0.3, 0.2, 0.15]]]], dtype=np.float32)
    images = []
    pixels = []
    for colors in ("#00ffff", (0, 255, 255), (0, 1, 1, 0.25)):
        group = actor.ellipsoid_slicer(lengths, colors=colors)
        with render_manager(group, width=3) as show_m:
            camera = show_m.screens[0].camera
            image = render_snapshot(show_m)
            point = _ellipsoid_field_pick(show_m, camera, (0, 0, 0), (0, 0, 0))
            rgb = image[point[1], point[0], :3]
            assert min(int(rgb[1]), int(rgb[2])) > 2 * int(rgb[0])
            images.append(image)
            pixels.append(rgb)
    np.testing.assert_allclose(images[0], images[1], atol=1)
    assert 0 < pixels[2][1] < 0.7 * pixels[0][1]
    assert 0 < pixels[2][2] < 0.7 * pixels[0][2]


@pytest.mark.parametrize("colors", [None, "#00ffff"])
def test_ellipsoid_slicer_inactive_direction_placeholders(colors):
    """Masked and zero-axis invalid frames stay warning-free and never pick."""
    lengths = np.full((5, 1, 1, 3), (0.15, 0.3, 0.1), dtype=np.float32)
    orientations = np.broadcast_to(np.eye(3), (5, 1, 1, 3, 3)).copy()
    mask = np.ones((5, 1, 1), dtype=bool)
    mask[1, 0, 0] = False
    lengths[1, 0, 0] = (np.nan, -np.inf, np.inf)
    lengths[2, 0, 0, 1] = 0
    lengths[3, 0, 0] = 0
    orientations[1:3] = np.nan
    orientations[1, 0, 0, 0, 0] = np.inf
    orientations[3] = np.inf
    originals = [value.copy() for value in (lengths, orientations, mask)]
    with (
        warnings.catch_warnings(),
        np.errstate(invalid="raise", divide="raise", over="raise"),
    ):
        warnings.simplefilter("error", RuntimeWarning)
        group = _create_chunked_ellipsoid_field(
            lengths,
            orientation_matrices=orientations,
            mask=mask,
            colors=colors,
            scale=0.8,
            actor_params={},
            max_glyphs=2,
        )
    with render_manager(
        group, target=(2, 0, 0), position=(2, 0, 10), width=7
    ) as show_m:
        camera = show_m.screens[0].camera
        image = render_snapshot(show_m)
        for x in range(5):
            voxel = (x, 0, 0)
            point = _ellipsoid_field_pixel(camera, voxel)
            if x in (0, 4):
                _ellipsoid_field_pick(show_m, camera, voxel, voxel)
                rgb = image[point[1], point[0], :3]
                assert rgb[1] > 2 * int(rgb[0])
                if colors is None:
                    assert rgb[1] > 2 * int(rgb[2])
                else:
                    assert rgb[2] > 2 * int(rgb[0])
            else:
                assert show_m.renderer.get_pick_info(point)["world_object"] is None
                assert image[point[1], point[0], :3].max() <= 5
    for value, original in zip((lengths, orientations, mask), originals, strict=True):
        np.testing.assert_array_equal(value, original)
