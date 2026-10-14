import numpy as np
import pytest

from fury.actor import (
    Group,
    contour_from_label,
    contour_from_roi,
    ellipsoid,
    get_slices,
    peaks_slicer,
    set_group_opacity,
    set_group_visibility,
    show_slices,
    tensor_slicer,
    volume_slicer,
)
from fury.actor.tests._helpers import (
    render_manager,
    render_snapshot,
    snapshot_extents,
    snapshot_mask,
)
from fury.lib import AffineTransform


def test_volume_slicer_comprehensive():
    """
    Test all major functionality of volume_slicer in one comprehensive
    test.
    """
    # Create test data - a simple 10x10x10 volume with gradient values
    data = np.random.rand(10, 10, 10)
    for i in range(10):
        data[i, :, :] = i / 10.0

    # Create test affine matrix with scaling and translation
    affine = np.eye(4)
    affine[0, 0] = 2.0  # Scale x-axis by 2
    affine[1, 1] = 0.5  # Scale y-axis by 0.5
    affine[2, 3] = 5.0  # Translate z-axis by 5

    # Test with all parameters specified
    actor = volume_slicer(
        data,
        affine=affine,
        value_range=(0.2, 0.8),
        opacity=0.7,
        interpolation="nearest",
        visibility=(True, False, True),
        initial_slices=(3, 5, 7),
    )

    # Verify basic properties
    assert isinstance(actor, Group)
    assert len(actor.children) == 3  # Should have 3 slice actors

    # Verify affine transform was applied correctly
    for child in actor.children:
        assert hasattr(child, "local")
        assert isinstance(child.local, AffineTransform)
        assert np.allclose(child.local.matrix, affine)

    # Verify visibility settings (only x and z should be visible)
    assert actor.children[0].visible is True  # x slice
    assert actor.children[1].visible is False  # y slice
    assert actor.children[2].visible is True  # z slice

    # Verify opacity
    for child in actor.children:
        assert child.material.opacity == pytest.approx(0.7)

    # Verify interpolation
    for child in actor.children:
        assert child.material.interpolation == "nearest"

    # Verify affine
    for child in actor.children:
        assert isinstance(child.local, AffineTransform)
        assert np.allclose(child.local.matrix, affine)


def test_peaks_slicer_basic_functionality():
    """Test basic functionality with minimal required inputs."""
    peak_dirs = np.random.rand(5, 5, 5, 3)  # 3D vector field
    result = peaks_slicer(peak_dirs)

    assert result is not None
    assert isinstance(result, Group)
    assert result.name == "VectorFieldSlicer"
    assert hasattr(result, "get_bounding_box")
    assert np.all(result.get_bounding_box() == np.array([[0, 0, 0], [4, 4, 4]]))
    vf = result.children[0]
    assert np.all(vf.cross_section == np.array([2, 2, 2]))
    assert vf.material.opacity == 1.0  # Default value


def test_peaks_slicer_with_affine_transform():
    """Test with affine transformation."""
    peak_dirs = np.random.rand(3, 3, 3, 3)
    affine = np.eye(4)
    affine[:3, 3] = [10, 20, 30]  # Translation

    result = peaks_slicer(peak_dirs, affine=affine)
    assert result is not None
    assert isinstance(result, Group)
    # Verify bounding box was transformed
    assert hasattr(result, "get_bounding_box")
    assert hasattr(result, "bounds")
    assert np.allclose(result.bounds[0], np.array([10, 20, 30]))
    assert np.allclose(result.bounds[1], np.array([12, 22, 32]))
    assert np.allclose(get_slices(result), np.array([11, 21, 31]))


def test_peaks_slicer_invalid_peak_dirs_shape():
    """Test with invalid peak directions shape."""
    with pytest.raises(ValueError):
        peaks_slicer(np.random.rand(3, 3))  # Not enough dimensions

    with pytest.raises(ValueError):
        peaks_slicer(np.random.rand(3, 3, 3, 3, 3, 3))  # Too many dimensions


def test_peaks_slicer_peak_values():
    """Test different peak_values configurations."""
    peak_dirs = np.random.rand(2, 2, 2, 3)

    # Test single float value
    result1 = peaks_slicer(peak_dirs, peak_values=0.5)

    # Test array of values
    result2 = peaks_slicer(peak_dirs, peak_values=np.random.rand(2, 2, 2))

    assert result1 is not None
    assert result2 is not None


def test_peaks_slicer_actor_types():
    """Test different actor_type options."""
    peak_dirs = np.random.rand(2, 2, 2, 3)

    result_thin = peaks_slicer(peak_dirs, actor_type="thin_line")
    result_thick = peaks_slicer(peak_dirs, actor_type="line", thickness=2.0)

    assert result_thin is not None
    assert result_thick is not None


def test_peaks_slicer_visual_properties():
    """Test visual properties (colors, opacity)"""
    peak_dirs = np.random.rand(2, 2, 2, 3)
    colors = np.random.rand(8, 3)

    result = peaks_slicer(
        peak_dirs, colors=colors, opacity=0.5, visibility=(True, False, True)
    )

    assert result is not None
    assert isinstance(result, Group)
    assert result.children[0].material.opacity == 0.5


def test_peaks_slicer_cross_section():
    """Test cross_section parameter."""
    peak_dirs = np.random.rand(2, 2, 2, 3)

    result_default = peaks_slicer(peak_dirs)

    assert result_default is not None

    with pytest.raises((ValueError, TypeError)):
        peaks_slicer(peak_dirs, cross_section=0.0)


@pytest.mark.parametrize("affine", [None, np.diag([1, 1, 1, 1])])
def test_contour_from_roi(affine):
    """Test contour_from_roi with optional affine transformation."""
    data = np.zeros((5, 5, 5), dtype=int)
    data[2, 2, 2] = 1

    contours = contour_from_roi(data, affine=affine)
    assert isinstance(contours, Group)
    assert len(contours.children) > 0
    actor = contours.children[0]
    assert np.allclose(actor.material.color[:3], [1, 0, 0])
    assert actor.material.opacity == pytest.approx(0.5)


@pytest.mark.parametrize(
    "colors,opacities",
    [
        (None, None),
        (np.array([[1, 0, 0], [0, 1, 0]]), np.array([0.8, 0.2])),
        (np.array([[0, 0, 1, 0.3]]), None),
    ],
)
def test_contour_from_label(colors, opacities):
    """
    Test contour_from_label with various color and opacity
    configurations.
    """
    data = np.zeros((5, 5, 5), dtype=int)
    data[1, 1, 1] = 1
    if colors is None or colors.shape[0] == 2:
        data[3, 3, 3] = 2

    contours = contour_from_label(data, colors=colors, opacities=opacities)
    assert isinstance(contours, Group)
    assert len(contours.children) == (colors.shape[0] if colors is not None else 2)

    if colors is not None and colors.shape[1] == 4:
        actor = contours.children[0].children[0]
        assert np.allclose(actor.material.color[:3], colors[0, :3])
        assert actor.material.opacity == pytest.approx(colors[0, 3])
    elif colors is not None:
        for i in range(len(contours.children)):
            actor = contours.children[i].children[0]
            assert np.allclose(actor.material.color[:3], colors[i])
            if opacities is not None:
                assert actor.material.opacity == pytest.approx(opacities[i])


@pytest.mark.parametrize(
    "colors,opacities,error_msg",
    [
        (np.array([[1, 0]]), None, "must have 3 or 4 channels"),
        (np.array([[1, 0, 0]]), np.array([0.5, 0.5]), "Incorrect opacity array shape"),
    ],
)
def test_contour_from_label_errors(colors, opacities, error_msg):
    """Test contour_from_label error conditions."""
    data = np.zeros((5, 5, 5), dtype=int)
    data[1, 1, 1] = 1

    with pytest.raises(ValueError, match=error_msg):
        contour_from_label(data, colors=colors, opacities=opacities)


def _tensor_pixel(camera, center):
    clip = camera.camera_matrix @ np.append(center, 1)
    ndc = clip[:2] / clip[3]
    return tuple(np.floor((ndc * (1, -1) + 1) * 128).astype(int))


def _tensor_pick(show_m, group, center, voxel):
    point = _tensor_pixel(show_m.screens[0].camera, center)
    pick = show_m.renderer.get_pick_info(point)
    child = pick["world_object"]
    assert child in group.children
    assert pick["voxel_index"] == voxel
    local_voxel = np.asarray(voxel) - child.voxel_origin
    assert pick["glyph_index"] == np.ravel_multi_index(
        tuple(local_voxel), child.field_shape
    )
    return point


@pytest.mark.parametrize("eigenframe", ["original", "reordered", "sign-flipped"])
def test_tensor_slicer_eigensystem(eigenframe):
    evals = np.array([[[[3.0, 1.0, 1.0]]]])
    evecs = np.eye(3)[None, None, None]
    if eigenframe == "reordered":
        permutation = [1, 2, 0]
        evals = evals[..., permutation]
        evecs = evecs[..., permutation]
    elif eigenframe == "sign-flipped":
        evecs = evecs * [-1, 1, 1]
    group = tensor_slicer(evals, evecs, norm=True, scale=0.6)
    assert isinstance(group, Group)
    np.testing.assert_allclose(
        group.children[0].get_bounding_box(),
        [[-0.6, -0.2, -0.2], [0.6, 0.2, 0.2]],
        atol=1e-7,
    )
    with render_manager(group, width=3) as show_m:
        image = render_snapshot(show_m)
        _tensor_pick(show_m, group, (0, 0, 0), (0, 0, 0))
        assert np.max(image[..., 1:3]) == 0
        assert show_m.renderer.get_pick_info((190, 128))["world_object"] is None
    extents = snapshot_extents(snapshot_mask(image))
    np.testing.assert_allclose(extents, [102, 34], atol=2)
    assert abs(extents[0] - 3 * extents[1]) <= 2

    reference = ellipsoid(
        [[0, 0, 0]],
        lengths=(0.6, 0.2, 0.2),
        colors=(np.sqrt(4 / 11), 0, 0),
        impostor=True,
    )
    with render_manager(reference, width=3) as show_m:
        np.testing.assert_allclose(render_snapshot(show_m), image, atol=2)

    mesh = ellipsoid(
        [[0, 0, 0]],
        lengths=(0.6, 0.2, 0.2),
        colors=(np.sqrt(4 / 11), 0, 0),
        impostor=False,
        phi=64,
        theta=64,
    )
    with render_manager(mesh, width=3) as show_m:
        mesh_mask = snapshot_mask(render_snapshot(show_m))
    mask = snapshot_mask(image)
    assert (
        np.count_nonzero(mask & mesh_mask) / np.count_nonzero(mask | mesh_mask) > 0.95
    )


def test_tensor_slicer_oblique_direction_colors():
    direction = np.array((1.0, 2.0, 2.0)) / 3
    second = np.array((-2.0, 1.0, 0.0)) / np.sqrt(5)
    orientation = np.column_stack((direction, second, np.cross(direction, second)))
    group = tensor_slicer(
        np.array([[[[3.0, 1.0, 1.0]]]]),
        orientation[None, None, None],
        scale=0.6,
    )
    reference = ellipsoid(
        [[0, 0, 0]],
        orientation_matrices=orientation,
        lengths=(0.6, 0.2, 0.2),
        colors=np.sqrt(4 / 11) * direction,
        impostor=True,
    )
    with render_manager(group, width=3) as show_m:
        image = render_snapshot(show_m)
        _tensor_pick(show_m, group, (0, 0, 0), (0, 0, 0))
    with render_manager(reference, width=3) as show_m:
        np.testing.assert_allclose(render_snapshot(show_m), image, atol=2)


@pytest.mark.parametrize("magnitude", [1e-300, 1e300])
def test_tensor_slicer_extreme_finite_eigenvalues(magnitude):
    evals = np.array([[[[3.0, 1.0, 1.0]]]]) * magnitude
    evecs = np.eye(3)[None, None, None]
    group = tensor_slicer(evals, evecs, scale=0.6)
    reference = ellipsoid(
        [[0, 0, 0]],
        lengths=(0.6, 0.2, 0.2),
        colors=(np.sqrt(4 / 11), 0, 0),
        impostor=True,
    )
    with render_manager(group, width=3) as show_m:
        image = render_snapshot(show_m)
        _tensor_pick(show_m, group, (0, 0, 0), (0, 0, 0))
    with render_manager(reference, width=3) as show_m:
        np.testing.assert_allclose(render_snapshot(show_m), image, atol=2)


@pytest.mark.parametrize("norm", [True, False])
def test_tensor_slicer_proportional_eigenvalues(norm):
    sizes = []
    for factor in (1, 2):
        evals = np.array([[[[3.0, 1.0, 1.0]]]]) * factor
        evecs = np.eye(3)[None, None, None]
        group = tensor_slicer(evals, evecs, norm=norm, scale=0.2)
        with render_manager(group, width=3) as show_m:
            sizes.append(snapshot_extents(snapshot_mask(render_snapshot(show_m))))
            _tensor_pick(show_m, group, (0, 0, 0), (0, 0, 0))
    np.testing.assert_allclose(sizes[1], sizes[0] * (1 if norm else 2), atol=2)


@pytest.mark.parametrize("colors", [None, (0, 1, 0)], ids=["FA", "explicit"])
@pytest.mark.parametrize("mask_dtype", [bool, np.int16, np.float64])
def test_tensor_slicer_background_and_colors(colors, mask_dtype):
    """
    Render numeric masks without changing eigenpairs or background omission.

    Parameters
    ----------
    colors : tuple or None
        Explicit color or FA-weighted orientation coloring.
    mask_dtype : type
        Numeric or boolean input mask dtype.

    Returns
    -------
    None
        Assertions check rendering, picking, and input ownership.
    """
    evals = np.array(
        [[1, 1, 1], [0, 0, 0], [-3, 1, 1], [np.nan] * 3, [3, 1, 1]],
        dtype=float,
    )[:, None, None, :]
    evecs = np.broadcast_to(np.eye(3), (5, 1, 1, 3, 3)).copy()
    evecs[3] = np.nan
    evecs[1] = np.nan
    evecs[2] = np.inf
    mask = np.ones((5, 1, 1), dtype=mask_dtype)
    mask[3] = False
    original_evals = evals.copy()
    original_evecs = evecs.copy()
    original_mask = mask.copy()
    group = tensor_slicer(
        evals,
        evecs,
        mask=mask,
        colors=colors,
        scale=0.3,
        visibility=(False, False, True),
    )
    with render_manager(
        group, position=(2, 0, 10), target=(2, 0, 0), width=6
    ) as show_m:
        image = render_snapshot(show_m)
        isotropic_pixel = _tensor_pick(show_m, group, (0, 0, 0), (0, 0, 0))
        _tensor_pick(show_m, group, (4, 0, 0), (4, 0, 0))
        px, py = isotropic_pixel
        if colors is None:
            assert not np.any(image[py - 3 : py + 4, px - 3 : px + 4, :3])
        else:
            assert image[py, px, 1] > 50
            assert image[py, px, 0] == image[py, px, 2] == 0
        for x in (1, 2, 3):
            point = _tensor_pixel(show_m.screens[0].camera, (x, 0, 0))
            assert show_m.renderer.get_pick_info(point)["world_object"] is None
            px, py = point
            assert not np.any(image[py - 5 : py + 6, px - 5 : px + 6, :3])
    np.testing.assert_array_equal(evals, original_evals)
    np.testing.assert_array_equal(evecs, original_evecs)
    np.testing.assert_array_equal(mask, original_mask)


@pytest.mark.parametrize("source", ["evals", "evecs"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_tensor_slicer_nonfinite_eigenpairs(source, invalid):
    evals = np.ones((1, 1, 1, 3))
    evecs = np.eye(3)[None, None, None].copy()
    if source == "evals":
        evals[0, 0, 0, 1] = invalid
    else:
        evecs[0, 0, 0, 1, 2] = invalid
    with pytest.raises(ValueError):
        tensor_slicer(evals, evecs)


@pytest.mark.parametrize("scale", [-1, np.nan, np.inf, -np.inf])
def test_tensor_slicer_invalid_scale(scale):
    with pytest.raises(ValueError):
        tensor_slicer(np.ones((1, 1, 1, 3)), np.eye(3)[None, None, None], scale=scale)


@pytest.mark.parametrize(
    ("evals_shape", "evecs_shape"),
    [
        ((3,), (3, 3)),
        ((2, 1, 1, 2), (2, 1, 1, 3, 3)),
        ((2, 1, 1, 3), (1, 1, 1, 3, 3)),
        ((2, 1, 1, 3), (2, 1, 1, 3)),
        ((2, 1, 1, 3), (2, 1, 1, 3, 2)),
        ((0, 1, 1, 3), (0, 1, 1, 3, 3)),
    ],
)
def test_tensor_slicer_invalid_eigensystem_shape(evals_shape, evecs_shape):
    with pytest.raises(ValueError):
        tensor_slicer(np.zeros(evals_shape), np.zeros(evecs_shape))


@pytest.mark.parametrize("mask_shape", [(2,), (1, 1, 1)])
@pytest.mark.parametrize("value", [1.0, np.nan])
def test_tensor_slicer_invalid_mask(mask_shape, value):
    """
    Reject broadcast masks before validating active eigenvalues.

    Parameters
    ----------
    mask_shape : tuple
        Broadcastable but incorrect spatial shape.
    value : float
        Finite or invalid eigenvalue.

    Returns
    -------
    None
        Assertions check early shape validation.
    """
    evals = np.full((2, 2, 2, 3), value)
    evecs = np.broadcast_to(np.eye(3), (2, 2, 2, 3, 3))
    mask = np.ones(mask_shape, dtype=bool)
    with pytest.raises(ValueError, match="Mask.*shape"):
        tensor_slicer(evals, evecs, mask=mask)


@pytest.mark.parametrize(
    ("magnitude", "dtype", "norm", "scale"),
    [
        (1e300, np.float64, False, 1),
        (1e38, np.float32, False, 10),
        (1, np.float64, True, 1e300),
        (1e300, np.float64, False, 1e100),
    ],
    ids=["raw-lengths", "float32-overflow", "normalized-scale", "float64-overflow"],
)
def test_tensor_slicer_unrepresentable_lengths(magnitude, dtype, norm, scale):
    """
    Reject finite eigenvalues whose scaled semi-axes exceed float32 range.

    Parameters
    ----------
    magnitude : float
        Multiplier for the original eigenvalues.
    dtype : type
        Input floating-point dtype.
    norm : bool
        Whether eigenvalues are normalized before scaling.
    scale : float
        Semi-axis multiplier.

    Returns
    -------
    None
        Assertions check the representability error.
    """
    evals = np.array([[[[3.0, 1.0, 1.0]]]], dtype=dtype) * magnitude
    evecs = np.eye(3)[None, None, None]
    with pytest.raises(ValueError, match="fit in float32"):
        tensor_slicer(evals, evecs, norm=norm, scale=scale)


@pytest.mark.parametrize(
    "affine",
    [
        np.eye(3),
        np.array([[1, 0, 0, np.nan], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        np.diag([1, np.inf, 1, 1]),
    ],
    ids=["shape", "nan", "infinite"],
)
def test_tensor_slicer_invalid_affine(affine):
    with pytest.raises(ValueError):
        tensor_slicer(np.ones((1, 1, 1, 3)), np.eye(3)[None, None, None], affine=affine)


def test_tensor_slicer_affine_slice_pick():
    evals = np.broadcast_to([3.0, 1.0, 1.0], (3, 2, 1, 3))
    evecs = np.broadcast_to(np.eye(3), (3, 2, 1, 3, 3))
    affine = np.array(
        [[2, 0.4, 0, 10.25], [0.3, 1.5, 0, -3.25], [0, 0, 1, 2], [0, 0, 0, 1]]
    )
    bounds = [[10.25, -3.25, 2], [14.65, -1.15, 2]]
    options = {
        "affine": affine,
        "scale": 0.12,
        "visibility": (True, False, False),
    }
    default = tensor_slicer(evals, evecs, **options)
    np.testing.assert_allclose(default.bounds, bounds)
    np.testing.assert_array_equal(get_slices(default), [12, -2, 2])
    group = tensor_slicer(evals, evecs, cross_section=(11, -2, 2), **options)
    np.testing.assert_allclose(group.bounds, bounds)
    np.testing.assert_array_equal(get_slices(group), [11, -2, 2])
    for child in group.children:
        np.testing.assert_allclose(child.bounds, bounds)

    with render_manager(
        group, position=(12, -2, 12), target=(12, -2, 2), width=8
    ) as show_m:
        initial = render_snapshot(show_m)
        assert np.max(initial[..., 1:3]) == 0
        # Disabled planes still snap matching center components.
        first = _tensor_pick(show_m, group, (11, -3.25, 2), (0, 0, 0))
        _tensor_pick(show_m, group, (11, -2, 2), (0, 1, 0))
        absent = _tensor_pixel(show_m.screens[0].camera, (12.25, -2.95, 2))
        assert show_m.renderer.get_pick_info(absent)["world_object"] is None

        show_slices(group, (12, -2, 2))
        moved = render_snapshot(show_m)
        np.testing.assert_array_equal(get_slices(group), [12, -2, 2])
        assert show_m.renderer.get_pick_info(first)["world_object"] is None
        point = _tensor_pick(show_m, group, (12, -2.95, 2), (1, 0, 0))
        _tensor_pick(show_m, group, (12, -2, 2), (1, 1, 0))
        assert not np.array_equal(initial, moved)

        set_group_opacity(group, 0.25)
        faded = render_snapshot(show_m)
        px, py = point
        assert 0 < faded[py, px, 0] < moved[py, px, 0]
        set_group_opacity(group, 1)
        set_group_visibility(group, False)
        assert not np.any(render_snapshot(show_m)[..., :3])
        assert show_m.renderer.get_pick_info(point)["world_object"] is None
        set_group_visibility(group, True)
        render_snapshot(show_m)
        _tensor_pick(show_m, group, (12, -2.95, 2), (1, 0, 0))

        for child in group.children:
            child.visible = False
        show_slices(group, (11, -2, 2))
        assert all(not child.visible for child in group.children)
        assert not np.any(render_snapshot(show_m)[..., :3])
        assert show_m.renderer.get_pick_info(first)["world_object"] is None

    with render_manager(
        default, position=(12, -2, 12), target=(12, -2, 2), width=8
    ) as show_m:
        render_snapshot(show_m)
        _tensor_pick(show_m, default, (12, -2.95, 2), (1, 0, 0))


@pytest.mark.parametrize(
    ("translation", "cross_section", "voxel"),
    [(0.25, 1, 0), (0.25, 2, 1), (0, 1, 0), (0, 1, 1)],
    ids=[
        "translated-first",
        "translated-second",
        "inclusive-half-spacing-left",
        "inclusive-half-spacing-right",
    ],
)
def test_tensor_slicer_affine_distance_boundary(translation, cross_section, voxel):
    evals = np.broadcast_to([3.0, 1.0, 1.0], (3, 1, 1, 3))
    evecs = np.broadcast_to(np.eye(3), (3, 1, 1, 3, 3))
    affine = np.diag([2.0, 1, 1, 1])
    affine[0, 3] = translation
    mask = np.zeros((3, 1, 1), dtype=bool)
    # Isolate each slot when both boundary neighbors snap to one location.
    mask[voxel, 0, 0] = True
    group = tensor_slicer(
        evals,
        evecs,
        affine=affine,
        mask=mask,
        scale=0.2,
        cross_section=(cross_section, 0, 0),
        visibility=(True, False, False),
        colors=(1, 0, 0),
    )
    center = (cross_section, 0, 0)
    with render_manager(
        group, position=(2, 0, 10), target=(2, 0, 0), width=6
    ) as show_m:
        assert np.max(render_snapshot(show_m)[..., 0]) > 50
        _tensor_pick(show_m, group, center, (voxel, 0, 0))
        show_slices(group, (4, 0, 0))
        assert not np.any(render_snapshot(show_m)[..., :3])
        point = _tensor_pixel(show_m.screens[0].camera, center)
        assert show_m.renderer.get_pick_info(point)["world_object"] is None


@pytest.mark.parametrize(("scale", "enable_picking"), [(0, True), (0.6, False)])
def test_tensor_slicer_zero_scale_and_disabled_picking(scale, enable_picking):
    evals = np.array([[[[3.0, 1.0, 1.0]]]])
    evecs = np.eye(3)[None, None, None]
    group = tensor_slicer(evals, evecs, scale=scale, enable_picking=enable_picking)
    with render_manager(group, width=3) as show_m:
        image = render_snapshot(show_m)
        if scale == 0:
            assert not np.any(image[..., :3])
        else:
            assert image[128, 128, 0] > max(image[128, 128, 1:3])
        assert show_m.renderer.get_pick_info((128, 128))["world_object"] is None


def test_tensor_slicer_shear_uses_diagonal_slice_distance():
    evals = np.broadcast_to([3.0, 1.0, 1.0], (3, 1, 1, 3))
    evecs = np.broadcast_to(np.eye(3), (3, 1, 1, 3, 3))
    affine = np.array([[2, 0, 0, 0.25], [1.6, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    group = tensor_slicer(
        evals,
        evecs,
        affine=affine,
        scale=0.12,
        cross_section=(1, 0, 0),
        visibility=(True, False, False),
        colors=(1, 0, 0),
    )
    with render_manager(
        group, position=(2, 1.6, 10), target=(2, 1.6, 0), width=6
    ) as show_m:
        image = render_snapshot(show_m)
        _tensor_pick(show_m, group, (1, 0, 0), (0, 0, 0))
        # Voxel 1 is outside half the diagonal spacing (1), but inside
        # half the column norm (about 1.28).
        absent = _tensor_pixel(show_m.screens[0].camera, (1, 1.6, 0))
        assert show_m.renderer.get_pick_info(absent)["world_object"] is None
        px, py = absent
        assert not np.any(image[py - 5 : py + 6, px - 5 : px + 6, :3])
        show_slices(group, (2, 0, 0))
        render_snapshot(show_m)
        _tensor_pick(show_m, group, (2, 1.6, 0), (1, 0, 0))


def test_tensor_slicer_fa_tie_chooses_first_eigenvector():
    evals = np.array([[[[3.0, 3.0, 1.0]]]])
    evecs = np.eye(3)[None, None, None]
    group = tensor_slicer(evals, evecs, scale=0.6)
    reference = ellipsoid(
        [[0, 0, 0]],
        lengths=(0.6, 0.6, 0.2),
        colors=(np.sqrt(4 / 19), 0, 0),
        impostor=True,
    )
    with render_manager(group, width=3) as show_m:
        image = render_snapshot(show_m)
        _tensor_pick(show_m, group, (0, 0, 0), (0, 0, 0))
        assert np.max(image[..., 1:3]) == 0
    with render_manager(reference, width=3) as show_m:
        np.testing.assert_allclose(render_snapshot(show_m), image, atol=2)
