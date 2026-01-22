import numpy as np
import pytest
from scipy.ndimage import generate_binary_structure

from fury.utils import (
    create_sh_basis_matrix,
    extract_surface_voxels,
    face_generation,
    generate_planar_uvs,
    get_lmax,
    get_n_coeffs,
    get_transformed_cube_bounds,
    voxel_mesh_by_object,
)


def test_generate_planar_uvs_basic_projections():
    """Test generate_planar_uvs with all three projection axes using simple geometry"""
    vertices = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    # XY projection
    xy_expected = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    assert np.allclose(generate_planar_uvs(vertices, axis="xy"), xy_expected)

    # XZ projection
    xz_expected = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    assert np.allclose(generate_planar_uvs(vertices, axis="xz"), xz_expected)

    # YZ projection
    yz_expected = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    assert np.allclose(generate_planar_uvs(vertices, axis="yz"), yz_expected)


def test_generate_planar_uvs_edge_cases():
    """Test generate_planar_uvs with various edge cases"""
    # All vertices same position
    with pytest.raises(ValueError):
        same_verts = np.array([[1.0, 1.0, 1.0]] * 3)
        generate_planar_uvs(same_verts)

    # Flat plane (zero range in one dimension)
    with pytest.raises(
        ValueError, match="Cannot generate UVs for flat geometry in the XY plane."
    ):
        flat_xy = np.array([[1.0, 2.0, 0.0], [2.0, 2.0, 0.0], [3.0, 2.0, 0.0]])
        generate_planar_uvs(flat_xy, axis="xy")

    with pytest.raises(
        ValueError, match="Cannot generate UVs for flat geometry in the XZ plane."
    ):
        flat_xz = np.array([[1.0, 0.0, 2.0], [2.0, 0.0, 2.0], [3.0, 0.0, 2.0]])
        generate_planar_uvs(flat_xz, axis="xz")

    with pytest.raises(
        ValueError, match="Cannot generate UVs for flat geometry in the YZ plane."
    ):
        flat_yz = np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 2.0], [0.0, 3.0, 2.0]])
        generate_planar_uvs(flat_yz, axis="yz")


def test_generate_planar_uvs_input_validation():
    """Test generate_planar_uvs input validation and error cases"""
    # Invalid axis
    with pytest.raises(ValueError, match="axis must be one of 'xy', 'xz', or 'yz'."):
        generate_planar_uvs(np.array([[1, 2, 3]]), axis="invalid")

    # Wrong array dimensions
    with pytest.raises(ValueError):
        generate_planar_uvs(np.array([1, 2, 3]))  # 1D array

    with pytest.raises(ValueError):
        generate_planar_uvs(np.array([[1, 2]]))  # Wrong shape

    with pytest.raises(ValueError):
        generate_planar_uvs(np.array([[1, 2, 3]]))  # Single vertex

    # Empty array
    with pytest.raises(ValueError):
        generate_planar_uvs(np.empty((0, 3)))


def test_generate_planar_uvs_numerical_stability():
    """Test generate_planar_uvs with numerical edge cases"""
    # Very small range
    small_range = np.array([[1.0, 2.0, 3.0], [1.0 + 1e-10, 2.0 + 1e-10, 3.0 + 1e-10]])
    result = generate_planar_uvs(small_range, axis="xy")
    assert not np.any(np.isnan(result))
    assert np.allclose(result, np.array([[0.0, 0.0], [1.0, 1.0]]))

    # Very large coordinates
    large_coords = np.array([[1e20, 2e20, 3e20], [2e20, 3e20, 4e20]])
    result = generate_planar_uvs(large_coords, axis="yz")
    assert not np.any(np.isnan(result))
    assert np.allclose(result, np.array([[0.0, 0.0], [1.0, 1.0]]))

    # Mixed positive and negative coordinates
    mixed_coords = np.array([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]])
    result = generate_planar_uvs(mixed_coords, axis="xz")
    assert np.allclose(result, np.array([[0.0, 0.0], [1.0, 1.0]]))


def test_get_lmax_standard_basis():
    """Test the standard basis type (default)."""
    assert get_lmax(3) == 1
    assert get_lmax(16) == 3
    assert get_lmax(24, basis_type="standard") == 4


def test_get_lmax_descoteaux07_basis():
    """Test the descoteaux07 basis type."""
    assert get_lmax(6, basis_type="descoteaux07") == 2
    assert get_lmax(28, basis_type="descoteaux07") == 6


def test_get_lmax_invalid_inputs():
    """Test invalid inputs raise ValueError."""
    with pytest.raises(ValueError):
        get_lmax(0)  # n_coeffs < 1
    with pytest.raises(ValueError):
        get_lmax(1.5)  # non-integer n_coeffs
    with pytest.raises(ValueError):
        get_lmax(10, basis_type="invalid")  # invalid basis_type


def test_get_n_coeffs_standard_basis():
    """Test the standard basis type (default)."""
    assert get_n_coeffs(1) == 4
    assert get_n_coeffs(3) == 16
    assert get_n_coeffs(4, basis_type="standard") == 25


def test_get_n_coeffs_descoteaux07_basis():
    """Test the descoteaux07 basis type."""
    assert get_n_coeffs(2, basis_type="descoteaux07") == 6
    assert get_n_coeffs(6, basis_type="descoteaux07") == 28


def test_get_n_coeffs_invalid_inputs():
    """Test invalid inputs raise ValueError."""
    with pytest.raises(ValueError):
        get_n_coeffs(-1)  # l_max < 0
    with pytest.raises(ValueError):
        get_n_coeffs(1.5)  # non-integer l_max
    with pytest.raises(ValueError):
        get_n_coeffs(2, basis_type="invalid")  # invalid basis_type


def test_get_n_coeffs_edge_cases():
    """Test edge cases for get_n_coeffs."""
    # l_max = 0 should give 1 coefficient
    assert get_n_coeffs(0) == 1
    assert get_n_coeffs(0, basis_type="descoteaux07") == 1

    # Test some higher values
    assert get_n_coeffs(5) == 36  # (5+1)^2 = 36
    assert get_n_coeffs(10) == 121  # (10+1)^2 = 121


def test_lmax_n_coeffs_inverse_relationship_standard():
    """Test that get_lmax and get_n_coeffs are inverse for standard basis."""
    test_lmax_values = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15]

    for l_max in test_lmax_values:
        # Test: get_lmax(get_n_coeffs(l_max)) == l_max
        n_coeffs = get_n_coeffs(l_max, basis_type="standard")
        recovered_lmax = get_lmax(n_coeffs, basis_type="standard")
        assert recovered_lmax == l_max, (
            f"Failed for l_max={l_max}: got {recovered_lmax}"
        )


def test_lmax_n_coeffs_inverse_relationship_descoteaux07():
    """Test that get_lmax and get_n_coeffs are inverse for descoteaux07 basis."""
    test_lmax_values = [0, 1, 2, 3, 4, 5, 6, 8, 10]

    for l_max in test_lmax_values:
        # Test: get_lmax(get_n_coeffs(l_max)) == l_max
        n_coeffs = get_n_coeffs(l_max, basis_type="descoteaux07")
        recovered_lmax = get_lmax(n_coeffs, basis_type="descoteaux07")
        assert recovered_lmax == l_max, (
            f"Failed for l_max={l_max}: got {recovered_lmax}"
        )


def test_n_coeffs_lmax_inverse_relationship_standard():
    """Test that get_n_coeffs and get_lmax are inverse for standard basis."""
    # Test with known valid n_coeffs values for standard basis: (l+1)^2
    test_n_coeffs_values = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121]

    for n_coeffs in test_n_coeffs_values:
        # Test: get_n_coeffs(get_lmax(n_coeffs)) == n_coeffs
        l_max = get_lmax(n_coeffs, basis_type="standard")
        recovered_n_coeffs = get_n_coeffs(l_max, basis_type="standard")
        assert recovered_n_coeffs == n_coeffs, (
            f"Failed for n_coeffs={n_coeffs}: got {recovered_n_coeffs}"
        )


def test_n_coeffs_lmax_inverse_relationship_descoteaux07():
    """Test that get_n_coeffs and get_lmax are inverse for descoteaux07 basis."""
    # Test with known valid n_coeffs values for descoteaux07 basis
    test_n_coeffs_values = [1, 6, 15, 28, 45, 66, 91, 120, 153]

    for n_coeffs in test_n_coeffs_values:
        # Test: get_n_coeffs(get_lmax(n_coeffs)) == n_coeffs
        l_max = get_lmax(n_coeffs, basis_type="descoteaux07")
        recovered_n_coeffs = get_n_coeffs(l_max, basis_type="descoteaux07")
        assert recovered_n_coeffs == n_coeffs, (
            f"Failed for n_coeffs={n_coeffs}: got {recovered_n_coeffs}"
        )


def test_both_functions_consistency():
    """Test consistency between both functions with various input combinations."""
    # Test that both functions handle basis_type parameter consistently
    l_max = 4

    # Standard basis
    n_coeffs_std = get_n_coeffs(l_max, basis_type="standard")
    recovered_lmax_std = get_lmax(n_coeffs_std, basis_type="standard")
    assert recovered_lmax_std == l_max

    # Descoteaux07 basis
    n_coeffs_desc = get_n_coeffs(l_max, basis_type="descoteaux07")
    recovered_lmax_desc = get_lmax(n_coeffs_desc, basis_type="descoteaux07")
    assert recovered_lmax_desc == l_max

    # Verify that the two basis types give different results
    assert n_coeffs_std != n_coeffs_desc


def test_create_sh_basis_matrix_input_validation():
    """Test invalid inputs raise ValueError."""
    # Invalid vertices (not a 2D array with shape (N, 3))
    with pytest.raises(ValueError):
        create_sh_basis_matrix(np.array([1, 2, 3]), 1)  # 1D array
    with pytest.raises(ValueError):
        create_sh_basis_matrix(np.array([[1, 2]]), 1)  # Shape (N, 2)

    # Invalid l_max (non-integer or negative)
    with pytest.raises(ValueError):
        create_sh_basis_matrix(np.array([[0, 0, 1]]), -1)
    with pytest.raises(ValueError):
        create_sh_basis_matrix(np.array([[0, 0, 1]]), 1.5)


def test_create_sh_basis_matrix_l_max_zero():
    """Test l_max=0 (only the constant SH term)."""
    vertices = np.array([[0, 0, 1], [1, 0, 0]])
    B = create_sh_basis_matrix(vertices, l_max=0)
    assert B.shape == (2, 1)  # (N_vertices, 1 coefficient)
    assert np.allclose(B, 1 / (2 * np.sqrt(np.pi)))  # Y_0^0 = 1/(2√π)


def test_create_sh_basis_matrix_basic_output_shape():
    """Verify output shape matches (N, (l_max+1)^2)."""
    vertices = np.random.randn(10, 3)  # 10 random vertices
    for l_max in [1, 2, 3]:
        B = create_sh_basis_matrix(vertices, l_max)
        assert B.shape == (10, (l_max + 1) ** 2)


def test_create_sh_basis_matrix_known_values():
    """Test against known SH values at specific points."""
    # North pole (theta=0, phi=undefined)
    vertices = np.array([[0, 0, 1]])
    B = create_sh_basis_matrix(vertices, l_max=1)

    # Expected values for l_max=1:
    # Y_0^0 = 1/(2√π)
    # Y_1^{-1} = 0 (due to sin(phi) term, but phi is undefined at pole)
    # Y_1^0 = √(3/4π)*cos(0) = √(3/4π)
    # Y_1^1 = 0 (due to cos(phi) term)
    expected = np.array([[1 / (2 * np.sqrt(np.pi)), 0, np.sqrt(3 / (4 * np.pi)), 0]])
    assert np.allclose(B, expected, atol=1e-6)


def test_get_transformed_cube_bounds_valid_input():
    """Test function with valid inputs returns correct bounds"""
    affine_matrix = np.eye(4)
    vertex1 = np.array([1, 2, 3])
    vertex2 = np.array([4, 5, 6])

    result = get_transformed_cube_bounds(affine_matrix, vertex1, vertex2)
    expected = [np.array([1, 2, 3]), np.array([4, 5, 6])]

    assert np.array_equal(result[0], expected[0])
    assert np.array_equal(result[1], expected[1])


def test_get_transformed_cube_bounds_invalid_vertex_dimensions():
    """Test function raises ValueError for non-3D vertices"""
    affine_matrix = np.eye(4)

    with pytest.raises(ValueError, match="must be 3D coordinates"):
        get_transformed_cube_bounds(
            affine_matrix, np.array([1, 2]), np.array([4, 5, 6])
        )

    with pytest.raises(ValueError, match="must be 3D coordinates"):
        get_transformed_cube_bounds(
            affine_matrix, np.array([1, 2, 3]), np.array([4, 5])
        )


def test_get_transformed_cube_bounds_invalid_matrix_shape():
    """Test function raises ValueError for non-4x4 matrix"""
    vertex1 = np.array([1, 2, 3])
    vertex2 = np.array([4, 5, 6])

    with pytest.raises(ValueError, match="must be a 4x4 numpy array"):
        get_transformed_cube_bounds(np.eye(3), vertex1, vertex2)

    with pytest.raises(ValueError, match="must be a 4x4 numpy array"):
        get_transformed_cube_bounds("not_a_matrix", vertex1, vertex2)


def test_get_transformed_cube_bounds_translation():
    """Test function correctly handles translation"""
    affine_matrix = np.array(
        [[1, 0, 0, 10], [0, 1, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]]
    )
    vertex1 = np.array([1, 2, 3])
    vertex2 = np.array([4, 5, 6])

    result = get_transformed_cube_bounds(affine_matrix, vertex1, vertex2)
    expected = [np.array([11, 22, 33]), np.array([14, 25, 36])]

    assert np.array_equal(result[0], expected[0])
    assert np.array_equal(result[1], expected[1])


def test_get_transformed_cube_bounds_scaling():
    """Test function correctly handles scaling"""
    affine_matrix = np.array([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 0], [0, 0, 0, 1]])
    vertex1 = np.array([1, 1, 1])
    vertex2 = np.array([2, 2, 2])

    result = get_transformed_cube_bounds(affine_matrix, vertex1, vertex2)
    expected = [np.array([2, 3, 4]), np.array([4, 6, 8])]

    assert np.array_equal(result[0], expected[0])
    assert np.array_equal(result[1], expected[1])


def test_get_transformed_cube_bounds_degenerate_case():
    """Test function handles single-point cube correctly"""
    affine_matrix = np.eye(4)
    vertex1 = np.array([5, 5, 5])
    vertex2 = np.array([5, 5, 5])

    result = get_transformed_cube_bounds(affine_matrix, vertex1, vertex2)
    expected = [np.array([5, 5, 5]), np.array([5, 5, 5])]

    assert np.array_equal(result[0], expected[0])
    assert np.array_equal(result[1], expected[1])


def test_extract_surface_voxels_basic_and_missing_label():
    """Test extract_surface_voxels for existing and missing labels."""

    volume = np.zeros((3, 3, 3), dtype=np.uint8)
    volume[1, 1, 1] = 1

    struct = generate_binary_structure(rank=3, connectivity=1)

    # Existing label should return surface coordinates and object mask
    surface_data = extract_surface_voxels(volume, 1, structuring_element=struct)
    assert surface_data is not None
    surface_coords, object_mask = surface_data

    assert surface_coords.shape == (1, 3)
    np.testing.assert_array_equal(surface_coords[0], np.array([1, 1, 1]))
    np.testing.assert_array_equal(object_mask, volume == 1)

    # Missing label should return None
    assert extract_surface_voxels(volume, 2, structuring_element=struct) is None


def test_face_generation_basic_axes_and_signs():
    """Test face_generation for simple coords with positive and negative signs."""

    coords = np.array([[0, 0, 0]], dtype=int)

    # Positive X face (axis=0, sign=+1)
    quads_pos = face_generation(coords, axis=0, sign=1)
    expected_pos = np.array([[[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]], dtype=int)

    assert quads_pos.shape == (1, 4, 3)
    np.testing.assert_array_equal(quads_pos, expected_pos)

    # Negative X face (axis=0, sign=-1) should flip winding
    quads_neg = face_generation(coords, axis=0, sign=-1)
    expected_neg = np.array([[[0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]]], dtype=int)

    assert quads_neg.shape == (1, 4, 3)
    np.testing.assert_array_equal(quads_neg, expected_neg)


def test_face_generation_broadcasting_and_invalid_coords():
    """Test face_generation broadcasting behavior and input validation."""

    # Broadcasting over multiple coordinates and axes
    coords = np.array([[0, 0, 0], [1, 1, 1]], dtype=int)
    axes = np.array([0, 1], dtype=int)
    signs = np.array([1, -1], dtype=int)

    quads = face_generation(coords, axis=axes, sign=signs)
    assert quads.shape == (2, 4, 3)

    # Single 1D coordinate should be accepted and reshaped
    quad_single = face_generation(np.array([0, 0, 0]), axis=0, sign=1)
    assert quad_single.shape == (1, 4, 3)

    # Invalid coord shape should raise ValueError
    with pytest.raises(ValueError, match=r"coords must have shape \(N, 3\)."):
        face_generation(np.array([[0, 0]]), axis=0, sign=1)


# Test cases for voxel_mesh_by_object
def test_voxel_mesh_by_object_input_validation():
    """Test invalid inputs raise appropriate errors."""
    # Invalid volume (not a 3D array)
    with pytest.raises(ValueError, match="volume must be a 3D numpy array"):
        from fury.utils import voxel_mesh_by_object

        voxel_mesh_by_object(np.array([[1, 2], [3, 4]]))

    # Invalid connectivity
    with pytest.raises(ValueError, match="connectivity must be one of"):
        from fury.utils import voxel_mesh_by_object

        voxel_mesh_by_object(np.ones((3, 3, 3)), connectivity=4)

    # Invalid spacing
    with pytest.raises(ValueError, match="spacing must be a tuple of 3 elements"):
        from fury.utils import voxel_mesh_by_object

        voxel_mesh_by_object(np.ones((3, 3, 3)), spacing=(1.0, 1.0))

    # Invalid triangulate parameter
    with pytest.raises(ValueError, match="triangulate must be a boolean value"):
        from fury.utils import voxel_mesh_by_object

        voxel_mesh_by_object(np.ones((3, 3, 3)), triangulate="yes")


def test_voxel_mesh_by_object_empty_volume():
    """Test that an empty volume (all zeros) returns an empty dictionary."""
    from fury.utils import voxel_mesh_by_object

    volume = np.zeros((5, 5, 5))
    result = voxel_mesh_by_object(volume)

    assert isinstance(result, dict)
    assert len(result) == 0


def test_voxel_mesh_by_object_single_voxel():
    """Test mesh generation for a single voxel."""
    from fury.utils import voxel_mesh_by_object

    volume = np.zeros((3, 3, 3))
    volume[1, 1, 1] = 1

    result = voxel_mesh_by_object(volume)

    assert len(result) == 1
    assert 1 in result
    assert "verts" in result[1]
    assert "faces" in result[1]

    # A single voxel should have 8 vertices (cube corners)
    assert result[1]["verts"].shape[0] == 8
    # A single voxel should have 6 quad faces (or 12 triangular faces if triangulated)
    assert result[1]["faces"].shape[0] == 6


def test_voxel_mesh_by_object_single_voxel_triangulated():
    """Test mesh generation for a single voxel with triangulation."""
    from fury.utils import voxel_mesh_by_object

    volume = np.zeros((3, 3, 3))
    volume[1, 1, 1] = 1

    result = voxel_mesh_by_object(volume, triangulate=True)

    assert len(result) == 1
    assert 1 in result

    # With triangulation, 6 quad faces become 12 triangular faces
    assert result[1]["faces"].shape[0] == 12
    # Each face should have 3 vertices (triangles)
    assert result[1]["faces"].shape[1] == 3


def test_voxel_mesh_by_object_multiple_objects():
    """Test mesh generation for multiple disconnected objects."""
    from fury.utils import voxel_mesh_by_object

    volume = np.zeros((10, 10, 10))
    # Object 1
    volume[2:4, 2:4, 2:4] = 1
    # Object 2 (disconnected)
    volume[6:8, 6:8, 6:8] = 1

    result = voxel_mesh_by_object(volume, connectivity=1)

    # Should have 2 separate objects
    assert len(result) == 2
    assert 1 in result
    assert 2 in result

    # Both objects should have vertices and faces
    for obj_id in [1, 2]:
        assert "verts" in result[obj_id]
        assert "faces" in result[obj_id]
        assert result[obj_id]["verts"].shape[0] > 0
        assert result[obj_id]["faces"].shape[0] > 0


def test_voxel_mesh_by_object_connectivity():
    """Test different connectivity options."""
    from fury.utils import voxel_mesh_by_object

    # Create a diagonal configuration
    volume = np.zeros((5, 5, 5))
    volume[1, 1, 1] = 1
    volume[2, 2, 2] = 1

    # With connectivity=1 (6-neighborhood), should be 2 separate objects
    result_6 = voxel_mesh_by_object(volume, connectivity=1)
    assert len(result_6) == 2

    # With connectivity=3 (26-neighborhood), should be 1 connected object
    result_26 = voxel_mesh_by_object(volume, connectivity=3)
    assert len(result_26) == 1


def test_voxel_mesh_by_object_spacing():
    """Test that spacing correctly scales the mesh vertices."""
    from fury.utils import voxel_mesh_by_object

    volume = np.zeros((3, 3, 3))
    volume[1, 1, 1] = 1

    # Test with different spacing
    spacing = (2.0, 3.0, 4.0)
    result = voxel_mesh_by_object(volume, spacing=spacing)

    verts = result[1]["verts"]

    # Check that vertices are scaled by the spacing
    # For a voxel at position (1,1,1), vertices should be in range
    # [1*spacing[i], 2*spacing[i]] for each dimension
    assert np.min(verts[:, 0]) >= 1.0 * spacing[0]
    assert np.max(verts[:, 0]) <= 2.0 * spacing[0]
    assert np.min(verts[:, 1]) >= 1.0 * spacing[1]
    assert np.max(verts[:, 1]) <= 2.0 * spacing[1]
    assert np.min(verts[:, 2]) >= 1.0 * spacing[2]
    assert np.max(verts[:, 2]) <= 2.0 * spacing[2]


def test_voxel_mesh_by_object_axis_ordering():
    """Ensure voxel coordinates are interpreted as (x, y, z) without swapping axes."""
    volume = np.zeros((2, 3, 4), dtype=np.uint8)
    volume[1, 2, 3] = 1  # voxel at x=1, y=2, z=3

    spacing = (2.0, 3.0, 5.0)
    result = voxel_mesh_by_object(volume, spacing=spacing)

    verts = result[1]["verts"]
    mins = np.min(verts, axis=0)
    maxs = np.max(verts, axis=0)

    expected_min = np.array([1, 2, 3], dtype=float) * np.array(spacing)
    expected_max = np.array([2, 3, 4], dtype=float) * np.array(spacing)

    assert np.allclose(mins, expected_min)
    assert np.allclose(maxs, expected_max)


def test_voxel_mesh_by_object_cube_block():
    """Test mesh generation for a 2x2x2 cube block."""
    from fury.utils import voxel_mesh_by_object

    volume = np.zeros((5, 5, 5))
    volume[1:3, 1:3, 1:3] = 1

    result = voxel_mesh_by_object(volume)

    assert len(result) == 1
    assert 1 in result

    # A 2x2x2 block has 8 voxels but shares internal faces
    # Only exterior faces should be generated
    verts = result[1]["verts"]
    faces = result[1]["faces"]

    assert verts.shape[0] > 0
    assert faces.shape[0] > 0

    # Vertices should be within the block bounds
    assert np.min(verts[:, 0]) >= 1.0
    assert np.max(verts[:, 0]) <= 3.0


def test_voxel_mesh_by_object_output_types():
    """Test that output arrays have correct dtypes."""
    from fury.utils import voxel_mesh_by_object

    volume = np.zeros((3, 3, 3))
    volume[1, 1, 1] = 1

    result = voxel_mesh_by_object(volume)

    assert result[1]["verts"].dtype == np.float32
    assert result[1]["faces"].dtype == np.int32


def test_voxel_mesh_by_object_face_orientation():
    """Test that faces are properly oriented (watertight mesh)."""
    from fury.utils import voxel_mesh_by_object

    volume = np.zeros((3, 3, 3))
    volume[1, 1, 1] = 1

    result = voxel_mesh_by_object(volume, triangulate=True)

    faces = result[1]["faces"]
    verts = result[1]["verts"]

    # Check that all face indices are valid
    assert np.all(faces >= 0)
    assert np.all(faces < len(verts))

    # Check that each face has 3 unique vertices
    for face in faces:
        assert len(np.unique(face)) == 3


def test_voxel_mesh_by_object_large_volume():
    """Test with a larger volume to ensure scalability."""
    from fury.utils import voxel_mesh_by_object

    # Create a sphere-like object
    volume = np.zeros((20, 20, 20))
    center = np.array([10, 10, 10])
    radius = 5

    for i in range(20):
        for j in range(20):
            for k in range(20):
                if np.linalg.norm(np.array([i, j, k]) - center) <= radius:
                    volume[i, j, k] = 1

    result = voxel_mesh_by_object(volume)

    assert len(result) == 1
    assert 1 in result

    # Should have a reasonable number of vertices and faces
    verts = result[1]["verts"]
    faces = result[1]["faces"]

    assert verts.shape[0] > 100  # Sphere surface should have many vertices
    assert faces.shape[0] > 100  # And many faces


def test_voxel_mesh_by_object_hollow_object():
    """Test mesh generation for a hollow object (only surface voxels)."""
    from fury.utils import voxel_mesh_by_object

    volume = np.zeros((7, 7, 7))
    # Create a hollow cube
    volume[2:5, 2:5, 2] = 1  # Bottom
    volume[2:5, 2:5, 4] = 1  # Top
    volume[2:5, 2, 2:5] = 1  # Front
    volume[2:5, 4, 2:5] = 1  # Back
    volume[2, 2:5, 2:5] = 1  # Left
    volume[4, 2:5, 2:5] = 1  # Right

    result = voxel_mesh_by_object(volume)

    assert len(result) == 1
    assert 1 in result

    # Should generate a mesh
    verts = result[1]["verts"]
    faces = result[1]["faces"]

    assert verts.shape[0] > 0
    assert faces.shape[0] > 0
