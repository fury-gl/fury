import numpy as np
import pytest

from fury.actor import (
    Group,
    Mesh,
    apply_affine_to_actor,
    apply_affine_to_group,
    data_slicer,
    ellipsoid_slicer,
    get_slices,
    line_projection,
    set_group_opacity,
    set_group_visibility,
    set_opacity,
    show_slices,
    vector_field_slicer,
)
from fury.actor.slicer import _create_chunked_ellipsoid_field
import fury.actor.utils as actor_utils
from fury.lib import (
    AffineTransform,
    Geometry,
    Material,
    RecursiveTransform,
    WorldObject,
)


@pytest.fixture
def actor():
    return Mesh(Geometry(), Material())


@pytest.fixture
def group_slicer():
    data = np.random.rand(10, 20, 30)
    return data_slicer(data)


@pytest.fixture
def group_line_projection():
    lines = [
        np.asarray([[0, 20, -2], [0, 20, 3]]),
        np.asarray([[0, 0, -2], [0, 0, 3]]),
        np.asarray([[0, -20, -2], [0, -20, 3]]),
    ]
    projection_z = line_projection(
        lines,
        plane=(0, 0, -1, 0),
        colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        thickness=10,
        outline_thickness=1,
        outline_color=(1, 1, 1),
    )
    projection_y = line_projection(
        lines,
        plane=(0, -1, 0, 0),
        colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        thickness=10,
        outline_thickness=1,
        outline_color=(1, 1, 1),
    )
    projection_x = line_projection(
        lines,
        plane=(-1, 0, 0, 0),
        colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        thickness=10,
        outline_thickness=1,
        outline_color=(1, 1, 1),
    )

    obj = Group(name="Slicer")
    obj.add(projection_z)
    obj.add(projection_y)
    obj.add(projection_x)
    return obj


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


def test_set_group_visibility_gfx_group(group_slicer):
    visibility = (True, False)
    with pytest.raises(ValueError):
        set_group_visibility(group_slicer, visibility)


def test_set_group_opacity_type_error():
    with pytest.raises(TypeError):
        set_group_opacity("not a group", 0.5)


def test_set_group_opacity_valid(group_slicer):
    set_group_opacity(group_slicer, 0.7)
    for child in group_slicer.children:
        assert round(child.material.opacity, 2) == 0.7


def test_set_group_opacity_with_gfx_group(group_slicer):
    set_group_opacity(group_slicer, 0.42)
    for child in group_slicer.children:
        assert round(child.material.opacity, 2) == 0.42


def test_set_opacity_type_error_for_invalid_actor():
    with pytest.raises(TypeError, match="actor must be an instance of WorldObject"):
        set_opacity("not an actor", 0.5)


def test_set_opacity_on_actor(actor):
    set_opacity(actor, 0.4)
    assert round(actor.opacity, 2) == 0.4


def test_set_opacity_on_gfx_group(group_slicer):
    set_opacity(group_slicer, 0.35)
    for child in group_slicer.children:
        assert round(child.material.opacity, 2) == 0.35


def test_set_opacity_invalid_value(actor):
    with pytest.raises(ValueError):
        set_opacity(actor, 1.5)


def test_get_slices_type_error():
    with pytest.raises(TypeError):
        get_slices("not a group")


@pytest.mark.parametrize("count", [1, 2, 4])
def test_slice_groups_reject_invalid_plane_counts(
    group_slicer, group_line_projection, count
):
    group = Group(name="Renamed planes")
    planes = (*group_slicer.children, group_line_projection.children[0])
    group.add(*planes[:count])
    with pytest.raises(ValueError, match="exactly 3"):
        get_slices(group)
    with pytest.raises(ValueError, match="exactly 3"):
        show_slices(group, (1, 2, 3))


def test_get_slices_valid(group_slicer):
    for i, child in enumerate(group_slicer.children):
        child.material.plane = (0, 0, 0, i * 10)
    result = get_slices(group_slicer)
    expected = np.array([0, 10, 20])
    assert np.array_equal(result, expected)


@pytest.mark.parametrize("factory", [vector_field_slicer, ellipsoid_slicer])
def test_get_slices_field_group(factory):
    field = np.ones((3, 3, 3, 3), dtype=np.float32)
    group = factory(field, cross_section=[1, 1, 1])
    group.name = "Renamed field"
    np.testing.assert_array_equal(get_slices(group), np.array([1, 1, 1]))


@pytest.mark.parametrize("factory", [vector_field_slicer, ellipsoid_slicer])
def test_show_slices_field_group(factory):
    field = np.ones((3, 3, 3, 3), dtype=np.float32)
    group = factory(field, cross_section=[0, 1, 2])
    group.name = "Arbitrary slice group"
    show_slices(group, [2, 2, 2])
    for child in group.children:
        np.testing.assert_array_equal(child.cross_section, np.array([2, 2, 2]))


@pytest.mark.parametrize("child_index", [0, 1, 2])
@pytest.mark.parametrize("invalid_child", ["material", "missing_material", "group"])
def test_slice_groups_validate_every_plane_child(
    group_slicer, child_index, invalid_child
):
    children = list(group_slicer.children)
    if invalid_child == "material":
        children[child_index] = Mesh(Geometry(), Material())
    elif invalid_child == "missing_material":
        children[child_index] = Mesh()
    else:
        children[child_index] = Group()
    group = Group(name="Unrelated name")
    group.add(*children)
    with pytest.raises(AttributeError, match="plane"):
        get_slices(group)
    with pytest.raises(AttributeError, match="plane"):
        show_slices(group, (1, 2, 3))


def test_show_slices_type_error():
    with pytest.raises(TypeError):
        show_slices("not a group", (1, 2, 3))


def test_show_slices_valid(group_slicer, group_line_projection):
    for child in group_slicer.children:
        child.material.plane = (1, 2, 3, 0)
    position = (10, 20, 30)
    show_slices(group_slicer, position)
    for i, child in enumerate(group_slicer.children):
        expected_plane = (1, 2, 3, position[i] + 1e-3)
        np.testing.assert_equal(
            child.material.plane, np.asarray(expected_plane, dtype=np.float32)
        )

    for child in group_line_projection.children:
        child.plane = (1, 2, 3, 0)
    position = (10, 20, 30)
    show_slices(group_line_projection, position)
    for i, child in enumerate(group_line_projection.children):
        expected_plane = (1, 2, 3, position[i] + 1e-3)
        np.testing.assert_equal(
            child.plane, np.asarray(expected_plane, dtype=np.float32)
        )


def test_show_slices_with_list(group_slicer):
    position = [5, 6, 7]
    show_slices(group_slicer, position)
    for i, child in enumerate(group_slicer.children):
        assert child.material.plane[-1] == position[i] + 1e-3


def test_ellipsoid_group_slices_preserve_hidden_chunks():
    lengths = np.full((2, 3, 4, 3), 0.3, dtype=np.float32)
    group = _create_chunked_ellipsoid_field(
        lengths,
        orientation_matrices=None,
        mask=None,
        colors=(1, 1, 1),
        actor_params={"cross_section": (0, 0, 0)},
        max_glyphs=5,
    )
    group.name = "Renamed ellipsoids"
    group.children[-1].visible = False
    show_slices(group, (1, 2, 3))
    np.testing.assert_array_equal(get_slices(group), (1, 2, 3))
    for child in group.children:
        np.testing.assert_array_equal(child.cross_section, (1, 2, 3))
    assert not group.children[-1].visible


@pytest.mark.parametrize("incomplete", [False, True])
@pytest.mark.parametrize("field_first", [False, True])
def test_slice_groups_reject_mixed_child_interfaces(
    group_slicer, incomplete, field_first
):
    fields = ellipsoid_slicer(np.ones((1, 1, 1, 3), dtype=np.float32))
    children = [fields.children[0], group_slicer.children[0]]
    if incomplete:
        children.append(Mesh(Geometry(), Material()))
    if not field_first:
        children.reverse()
    group = Group()
    group.add(*children)
    with pytest.raises(AttributeError, match="cross_section"):
        get_slices(group)
    with pytest.raises(AttributeError, match="cross_section"):
        show_slices(group, (0, 0, 0))


def test_empty_slice_group():
    group = Group()
    with pytest.raises(ValueError, match="at least one"):
        get_slices(group)
    with pytest.raises(ValueError, match="at least one"):
        show_slices(group, (0, 0, 0))


def test_plane_slices_route_each_child(group_slicer, group_line_projection):
    group = Group(name="Renamed planes")
    group.add(
        group_slicer.children[0],
        group_line_projection.children[1],
        group_slicer.children[2],
    )
    show_slices(group, (2, 4, 6))
    np.testing.assert_allclose(get_slices(group), np.array((2, 4, 6)) + 1e-3)


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


@pytest.mark.parametrize("sync_cpu", [True, False])
def test_read_buffer_requires_underlying_wgpu_buffer(group_line_projection, sync_cpu):
    actor_obj = group_line_projection.children[0]
    buffer = actor_obj.geometry.positions

    with pytest.raises(AttributeError) as exc_info:
        actor_utils.read_buffer(buffer, sync_cpu=sync_cpu)
    assert "size" in str(exc_info.value)
    assert "Buffer" in str(exc_info.value)


def test_read_buffer_rejects_non_buffer_instance():
    with pytest.raises(ValueError, match="Expected a wgpu.Buffer instance."):
        actor_utils.read_buffer(np.zeros((2, 3), dtype=np.float32))
