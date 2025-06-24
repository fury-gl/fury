import numpy as np
import pytest

from fury.actor import slicer
from fury.lib import AffineTransform, Group, Mesh, RecursiveTransform, WorldObject
from fury.utils import (
    apply_affine_to_actor,
    apply_affine_to_group,
    create_sh_basis_matrix,
    generate_planar_uvs,
    get_lmax,
    get_slices,
    set_group_opacity,
    set_group_visibility,
    show_slices,
)


@pytest.fixture
def actor():
    return Mesh()


@pytest.fixture
def group_slicer():
    data = np.random.rand(10, 20, 30)
    return slicer(data)


# Test cases for set_group_visibility
def test_set_group_visibility_type_error():
    with pytest.raises(TypeError):
        set_group_visibility("not a group", True)


def test_set_group_visibility_single_bool():
    group = Group()
    group.visible = False
    set_group_visibility(group, True)
    assert group.visible is True


def test_set_group_visibility_list(group_slicer):
    visibility = [True, False, True]
    set_group_visibility(group_slicer, visibility)
    for actor, vis in zip(group_slicer.children, visibility, strict=False):
        assert actor.visible == vis


def test_set_group_visibility_tuple(group_slicer):
    visibility = (False, True, False)
    set_group_visibility(group_slicer, visibility)
    for actor, vis in zip(group_slicer.children, visibility, strict=False):
        assert actor.visible == vis


# Test cases for set_opacity
def test_set_opacity_type_error():
    with pytest.raises(TypeError):
        set_group_opacity("not a group", 0.5)


def test_set_opacity_valid(group_slicer):
    set_group_opacity(group_slicer, 0.7)
    for child in group_slicer.children:
        assert round(child.material.opacity, 2) == 0.7


# Test cases for get_slices
def test_get_slices_type_error():
    with pytest.raises(TypeError):
        get_slices("not a group")


def test_get_slices_value_error(actor):
    group = Group()
    group.add(actor, Mesh())
    with pytest.raises(ValueError):
        get_slices(group)


def test_get_slices_attribute_error(actor):
    group = Group()
    group.add(actor, Mesh(), Mesh())
    with pytest.raises(AttributeError):
        get_slices(group)


def test_get_slices_valid(group_slicer):
    for i, child in enumerate(group_slicer.children):
        child.material.plane = (0, 0, 0, i * 10)
    result = get_slices(group_slicer)
    expected = np.array([0, 10, 20])
    assert np.array_equal(result, expected)


# Test cases for show_slices
def test_show_slices_type_error():
    with pytest.raises(TypeError):
        show_slices("not a group", (1, 2, 3))


def test_show_slices_valid(group_slicer):
    for child in group_slicer.children:
        child.material.plane = (1, 2, 3, 0)
    position = (10, 20, 30)
    show_slices(group_slicer, position)
    for i, child in enumerate(group_slicer.children):
        expected_plane = (1, 2, 3, position[i])
        np.testing.assert_equal(child.material.plane, expected_plane)


def test_show_slices_with_list(group_slicer):
    position = [5, 6, 7]
    show_slices(group_slicer, position)
    for i, child in enumerate(group_slicer.children):
        assert child.material.plane[-1] == position[i]


def test_apply_affine_to_actor_valid_input():
    """Test apply_affine_to_actor with valid inputs."""
    # Create a mock actor
    actor = WorldObject()
    original_local = actor.local
    original_world = actor.world

    # Create test affine matrix
    affine = np.eye(4)
    affine[:3, :3] = 2 * np.eye(3)  # Scale by 2

    apply_affine_to_actor(actor, affine)

    # Check that the transforms were updated
    assert actor.local != original_local
    assert actor.world != original_world
    assert isinstance(actor.local, AffineTransform)
    assert isinstance(actor.world, RecursiveTransform)
    assert np.allclose(actor.local.matrix, affine)


def test_apply_affine_to_actor_invalid_actor():
    """Test apply_affine_to_actor with invalid actor type."""
    with pytest.raises(TypeError, match="actor must be an instance of WorldObject"):
        apply_affine_to_actor("not_an_actor", np.eye(4))


def test_apply_affine_to_actor_invalid_affine():
    """Test apply_affine_to_actor with invalid affine matrix."""
    actor = WorldObject()

    with pytest.raises(ValueError):
        # Wrong shape
        apply_affine_to_actor(actor, np.eye(3))

    with pytest.raises(ValueError):
        # Not a numpy array
        apply_affine_to_actor(
            actor, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )


def test_apply_affine_to_group_valid_input(group_slicer):
    """Test apply_affine_to_group with valid inputs."""

    # Store original transforms
    original_transforms = [(c.local, c.world) for c in group_slicer.children]

    # Create test affine matrix
    affine = np.eye(4)
    affine[0, 3] = 10  # Translate in x by 10

    apply_affine_to_group(group_slicer, affine)

    # Check that all children were updated
    for child, (orig_local, orig_world) in zip(
        group_slicer.children, original_transforms, strict=False
    ):
        assert child.local != orig_local
        assert child.world != orig_world
        assert isinstance(child.local, AffineTransform)
        assert isinstance(child.world, RecursiveTransform)
        assert np.allclose(child.local.matrix, affine)


def test_apply_affine_to_group_empty_group():
    """Test apply_affine_to_group with an empty group."""
    group = Group()

    # This should not raise any exceptions
    apply_affine_to_group(group, np.eye(4))


def test_apply_affine_to_group_invalid_group():
    """Test apply_affine_to_group with invalid group type."""
    with pytest.raises(TypeError, match="group must be an instance of Group"):
        apply_affine_to_group("not_a_group", np.eye(4))


def test_affine_transform_properties():
    """Test that the affine transform properties are correctly set."""
    actor = WorldObject()
    affine = np.eye(4)
    affine[1, 1] = 3.0  # Scale y by 3

    apply_affine_to_actor(actor, affine)

    # Check transform properties
    assert actor.local.state_basis == "matrix"
    assert actor.local.is_camera_space == int(True)
    assert np.allclose(actor.local.matrix, affine)

    # Check recursive transform wraps the affine transform
    assert isinstance(actor.world, RecursiveTransform)
    assert actor.world.own == actor.local


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
