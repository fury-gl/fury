import logging
import math

import numpy as np
import pytest

from fury import actor, window
from fury.actor.curved import (
    _estimate_streamtube_buffer_size,
    _split_streamtube_lines,
)
from fury.actor.tests._helpers import (
    CENTERS,
    LINES,
    PBR_MATERIAL_TYPES,
    assert_rejects_pbr_params_on_phong,
    assert_supports_pbr,
    render_manager,
    render_snapshot,
    snapshot_extents,
    snapshot_mask,
    validate_actors,
)
from fury.lib import BufferUsage
from fury.material import (
    StreamlinesMaterial,
    _StreamlineBakedMaterial,
    _StreamtubeBakedMaterial,
)


def test_sphere():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="sphere", impostor=False)


def test_cylinder():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="cylinder")


def test_cone():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="cone")


@pytest.mark.parametrize("impostor", [False, True], ids=["mesh", "impostor"])
@pytest.mark.parametrize("broadcast", ["vector", "singleton", "per-glyph"])
def test_ellipsoid(impostor, broadcast):
    centers = np.array([[0, 0, 0], [2, 0, 0]])
    lengths = np.array([1, 0.5, 0.5])
    axes = np.eye(3)
    if broadcast == "singleton":
        lengths = lengths[None]
        axes = axes[None]
    elif broadcast == "per-glyph":
        lengths = np.broadcast_to(lengths, (2, 3))
        axes = np.broadcast_to(axes, (2, 3, 3))
    obj = actor.ellipsoid(
        centers, lengths=lengths, orientation_matrices=axes, impostor=impostor
    )
    with render_manager(obj) as show_m:
        image = render_snapshot(show_m)
        for x in (128, 192):
            assert image[128, x, 0] > image[128, x, 1]
            assert image[128, x, 0] > image[128, x, 2]
        obj.visible = False
        assert not snapshot_mask(render_snapshot(show_m)).any()


def test_streamtube():
    lines = [np.array([[0, 0, 0], [1, 1, 1]])]
    colors = np.array([[1, 0, 0]])
    scene = window.Scene()

    tube_actor = actor.streamtube(lines=lines, colors=colors)
    scene.add(tube_actor)

    arr = window.snapshot(scene=scene, fname=None, return_array=True)
    report = window.analyze_snapshot(arr, find_objects=True)
    assert report.objects >= 1

    mean_r, mean_g, mean_b, _ = np.mean(arr.reshape(-1, arr.shape[2]), axis=0)
    assert mean_r > mean_g and mean_r > mean_b

    middle_pixel = arr[arr.shape[0] // 2, arr.shape[1] // 2]
    r, g, b, a = middle_pixel
    assert r > g and r > b


def test_streamtube_gpu_geometry_and_buffers():
    """GPU streamtube: geometry, buffers, and material state consistency."""
    lines = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.5]], dtype=np.float32),
        np.array([[0.5, -0.5, 0.2], [0.5, 0.5, 0.8]], dtype=np.float32),
    ]
    colors = np.array([[1.0, 0.0, 0.0, 0.6], [0.0, 1.0, 0.0, 0.4]], dtype=np.float32)
    radius = 0.15
    segments = 5

    mesh = actor.streamtube(
        lines,
        colors=colors,
        opacity=0.75,
        radius=radius,
        segments=segments,
        end_caps=True,
        backend="gpu",
    )

    assert isinstance(mesh.material, _StreamtubeBakedMaterial)
    assert mesh.n_lines == len(lines)

    line_lengths = np.array([line.shape[0] for line in lines], dtype=np.uint32)
    assert np.array_equal(mesh.line_lengths, line_lengths)
    assert mesh.max_line_length == int(line_lengths.max())
    assert mesh.tube_sides == segments
    assert mesh.end_caps is True

    vertices_per_line = line_lengths * segments + 2
    expected_vertex_offsets = np.zeros_like(line_lengths)
    if len(lines) > 1:
        expected_vertex_offsets[1:] = np.cumsum(
            vertices_per_line[:-1], dtype=np.uint64
        ).astype(np.uint32)
    assert np.array_equal(mesh.vertex_offsets, expected_vertex_offsets)

    segments_per_line = np.maximum(line_lengths - 1, 0)
    triangles_per_line = segments_per_line * segments * 2 + segments * 2
    expected_triangle_offsets = np.zeros_like(line_lengths)
    if len(lines) > 1:
        expected_triangle_offsets[1:] = np.cumsum(
            triangles_per_line[:-1], dtype=np.uint64
        ).astype(np.uint32)
    assert np.array_equal(mesh.triangle_offsets, expected_triangle_offsets)

    total_vertices = int(vertices_per_line.astype(np.uint64).sum())
    total_triangles = int(triangles_per_line.astype(np.uint64).sum())
    assert mesh.geometry.positions.data.shape == (total_vertices, 3)
    assert mesh.geometry.indices.data.shape == (total_triangles, 3)
    assert mesh.geometry.colors.data.shape == (total_vertices, 3)

    reshaped_line_buffer = mesh.line_buffer.data.reshape(
        len(lines), mesh.max_line_length, 3
    )
    expected_line_data = np.zeros_like(reshaped_line_buffer)
    for idx, line in enumerate(lines):
        expected_line_data[idx, : line.shape[0]] = line
    assert np.allclose(reshaped_line_buffer, expected_line_data)

    expected_colors = colors[:, :3]
    assert np.allclose(mesh.line_colors, expected_colors)
    assert np.allclose(mesh.color_buffer.data, expected_colors)
    assert mesh.color_components == 3

    assert np.isclose(mesh.material.radius, radius)
    assert mesh.material.segments == segments
    assert mesh.material.end_caps is True
    assert mesh.material.line_count == mesh.n_lines


def test_streamtube_gpu_color_broadcast_and_material_flags():
    """GPU streamtube: color broadcasting and material flags."""
    lines = [
        np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 1.5, 0.5]], dtype=np.float32),
        np.array([[1.0, 0.0, 0.0], [1.5, 0.5, 0.5], [2.0, 1.0, 1.0]], dtype=np.float32),
    ]
    base_color = (0.2, 0.4, 0.6)

    mesh = actor.streamtube(
        lines,
        colors=base_color,
        segments=4,
        end_caps=False,
        enable_picking=False,
        backend="gpu",
    )

    expected_colors = np.tile(np.asarray(base_color, dtype=np.float32), (len(lines), 1))
    assert np.allclose(mesh.line_colors, expected_colors)
    assert np.allclose(mesh.color_buffer.data, expected_colors)
    assert mesh.color_components == 3

    assert isinstance(mesh.material, _StreamtubeBakedMaterial)
    assert mesh.material.pick_write is False
    assert mesh.material.flat_shading is False
    assert mesh.material.end_caps is False
    assert mesh.material.segments == 4


def test_streamtube_gpu_invalid_inputs():
    """GPU streamtube: invalid inputs raise informative errors."""
    line_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    line_b = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="material='phong' only"):
        actor.streamtube(
            [line_a], colors=(1.0, 0.0, 0.0), backend="gpu", material="basic"
        )

    with pytest.raises(
        ValueError, match=r"first dimension must be 1 or \d+ \(number of lines\)"
    ):
        actor.streamtube(
            [line_a, line_b],
            colors=np.ones((3, 3), dtype=np.float32),
            backend="gpu",
        )

    with pytest.raises(ValueError, match="must have 3 or 4 channels"):
        actor.streamtube(
            [line_a],
            colors=np.array([1.0, 0.5], dtype=np.float32),
            backend="gpu",
        )


def test_streamtube_buffer_helpers_and_split_ratio():
    """
    Buffer estimation and ratio-based splitting with device buffer
    limits.
    """

    def _make_lines(lengths):
        return [np.zeros((ln, 3), dtype=np.float32) for ln in lengths]

    # Buffer estimation sums all allocations
    lengths = np.array([2, 3], dtype=np.uint32)
    segments = 4
    end_caps = True
    color_components = 3

    total_vertices = (lengths * segments + 2).sum()
    total_triangles = ((lengths - 1) * segments * 2 + segments * 2).sum()
    n_lines = len(lengths)
    max_len = lengths.max()

    expected_total = (
        n_lines * max_len * 3 * 4
        + n_lines * 4
        + n_lines * color_components * 4
        + n_lines * 4
        + n_lines * 4
        + total_vertices * 3 * 4
        + total_vertices * 3 * 4
        + total_vertices * color_components * 4
        + total_triangles * 3 * 4
    )
    assert (
        _estimate_streamtube_buffer_size(lengths, segments, end_caps, color_components)
        == expected_total
    )

    # Ratio-based batch count
    line_lengths = [5, 5, 5, 5, 5, 5]
    lines = _make_lines(line_lengths)
    total_needed = _estimate_streamtube_buffer_size(
        np.array(line_lengths), segments=3, end_caps=True, color_components=3
    )
    max_buffer = total_needed // 3  # bytes, force ~3 batches
    batches = _split_streamtube_lines(
        lines,
        segments=3,
        end_caps=True,
        color_components=3,
        max_buffer_size=max_buffer,
    )
    expected_batches = math.ceil(total_needed / max_buffer)
    assert len(batches) == expected_batches
    assert sum(len(batch) for batch in batches) == len(lines)

    with pytest.raises(
        ValueError,
        match="Streamtube data for a single line exceeds the available buffer",
    ):
        _split_streamtube_lines(
            _make_lines([20]),
            segments=8,
            end_caps=True,
            color_components=3,
            max_buffer_size=1,  # deliberately tiny
        )

    # Integration sanity checks on both backends
    colors = np.eye(len(line_lengths), 3, dtype=np.float32)
    actor_cpu = actor.streamtube(
        lines,
        colors=colors,
        backend="cpu",
        segments=3,
        radius=0.1,
        end_caps=True,
    )
    assert actor_cpu is not None

    actor_gpu = actor.streamtube(
        lines,
        colors=colors,
        backend="gpu",
        segments=3,
        radius=0.1,
        end_caps=True,
    )
    assert actor_gpu is not None


def test_streamlines_roi_metadata_and_reset():
    """Streamlines ROI mask toggles baked material and restores buffers."""
    lines = [
        np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32),
    ]
    roi_mask = np.ones((2, 3, 4), dtype=np.uint8)

    wobj = actor.streamlines(lines, colors=(0.2, 0.2, 0.2), roi_mask=roi_mask)

    assert np.array_equal(wobj._line_lengths, np.array([2, 2], dtype=np.uint32))
    assert np.array_equal(wobj._line_offsets, np.array([0, 3], dtype=np.uint32))
    assert isinstance(wobj.material, _StreamlineBakedMaterial)
    assert wobj.material.roi_enabled is True
    assert wobj.material.roi_dim == roi_mask.shape
    assert np.allclose(wobj.roi_origin, (0.0, 0.0, 0.0))
    assert not np.isfinite(wobj.geometry.positions.data).all()

    wobj.roi_mask = None
    assert isinstance(wobj.material, StreamlinesMaterial)
    assert wobj.material.roi_enabled is False
    assert wobj._needs_gpu_update is False
    assert np.allclose(
        wobj.geometry.positions.data,
        wobj._input_positions_array.reshape(wobj.geometry.positions.data.shape),
        equal_nan=True,
    )


def test_streamlines_roi_origin_updates_needs_update_flag():
    """Changing ROI origin sets compute update when a mask is present."""
    lines = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]], dtype=np.float32),
    ]
    roi_mask = np.ones((3, 3, 3), dtype=np.uint8)

    wobj = actor.streamlines(
        lines, colors=(0.1, 0.1, 0.1), roi_mask=roi_mask, roi_origin=(1.0, 2.0, 3.0)
    )

    assert np.allclose(wobj.roi_origin, (1.0, 2.0, 3.0))
    wobj._needs_gpu_update = False
    wobj.roi_origin = (2.5, -1.0, 0.5)
    assert np.allclose(wobj.roi_origin, (2.5, -1.0, 0.5))
    assert wobj._needs_gpu_update is True


def test_streamlines_helper_populates_buffers_without_roi():
    """Actor helper populates metadata buffers when no ROI is provided."""
    lines = [
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        np.array([[1.0, 0.0, 0.0], [2.0, 1.0, 1.0]], dtype=np.float32),
    ]

    wobj = actor.streamlines(lines, colors=(1.0, 0.0, 0.0))

    assert isinstance(wobj.material, StreamlinesMaterial)
    assert np.array_equal(
        wobj._line_lengths_buffer.data, wobj._line_lengths.astype(np.uint32)
    )
    assert np.array_equal(
        wobj._line_offsets_buffer.data, wobj._line_offsets.astype(np.uint32)
    )
    assert np.allclose(
        wobj.geometry.positions.data,
        wobj._input_positions_array.reshape(wobj.geometry.positions.data.shape),
        equal_nan=True,
    )


def test_streamlines_filtered_streamlines_and_copy_src_usage():
    """filtered_streamlines requests readback from a COPY_SRC buffer."""
    lines = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32),
    ]
    wobj = actor.streamlines(lines, colors=(1.0, 0.0, 0.0))

    assert wobj.geometry.positions._wgpu_usage & BufferUsage.COPY_SRC

    with pytest.raises(AttributeError) as exc_info:
        wobj.filtered_streamlines()
    assert "size" in str(exc_info.value)
    assert "Buffer" in str(exc_info.value)


def test_streamtube_geometry_min_max_bounds():
    """Test that streamtube geometry stores min/max points correctly."""
    lines = [
        np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], dtype=np.float32),
        np.array([[-5.0, -2.0, -1.0], [0.5, 1.0, 1.5]], dtype=np.float32),
        np.array([[3.0, 8.0, 4.0], [4.0, 10.0, 5.0]], dtype=np.float32),
    ]

    mesh = actor.streamtube(
        lines, colors=(1.0, 0.0, 0.0), radius=0.1, segments=6, backend="gpu"
    )

    positions = mesh.geometry.positions.data

    expected_min = np.array([-5.0, -2.0, -1.0], dtype=np.float32)
    expected_max = np.array([4.0, 10.0, 6.0], dtype=np.float32)

    actual_min = positions[0]
    actual_max = positions[1]

    assert np.allclose(actual_min, expected_min), (
        f"Min point mismatch: expected {expected_min}, got {actual_min}"
    )
    assert np.allclose(actual_max, expected_max), (
        f"Max point mismatch: expected {expected_max}, got {actual_max}"
    )


def test_cone_per_instance_geometry():
    """Test cone actor with per-instance radii and height."""
    centers = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Per-instance radii and height
    radii = np.array([0.3, 0.5, 0.7])
    heights = np.array([1.0, 1.5, 2.0])
    cone_actor = actor.cone(centers=centers, colors=colors, radii=radii, height=heights)
    assert cone_actor.prim_count == 3

    # Verify geometry varies: per-instance vertex extents differ
    verts = cone_actor.geometry.positions.view
    n_verts_per = len(verts) // 3
    extents = []
    for i in range(3):
        chunk = verts[i * n_verts_per : (i + 1) * n_verts_per] - centers[i]
        extents.append(chunk[:, 0].max() - chunk[:, 0].min())
    # Larger radii should yield wider extents
    assert extents[0] < extents[1] < extents[2]

    # Scalar (backward compat)
    cone_actor_scalar = actor.cone(
        centers=centers, colors=colors, radii=0.5, height=1.0
    )
    assert cone_actor_scalar.prim_count == 3

    # Wrong-size array raises ValueError
    with pytest.raises(ValueError, match="radii"):
        actor.cone(
            centers=centers,
            colors=colors,
            radii=np.array([0.3, 0.5]),
        )

    with pytest.raises(ValueError, match="height"):
        actor.cone(
            centers=centers,
            colors=colors,
            height=np.array([1.0, 1.5]),
        )


def test_cylinder_per_instance_geometry():
    """Test cylinder actor with per-instance radii and height."""
    centers = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Per-instance radii and height
    radii = np.array([0.3, 0.5, 0.7])
    heights = np.array([1.0, 1.5, 2.0])
    cyl_actor = actor.cylinder(
        centers=centers, colors=colors, radii=radii, height=heights
    )
    assert cyl_actor.prim_count == 3

    # Verify geometry varies: per-instance vertex extents differ
    verts = cyl_actor.geometry.positions.view
    n_verts_per = len(verts) // 3
    extents = []
    for i in range(3):
        chunk = verts[i * n_verts_per : (i + 1) * n_verts_per] - centers[i]
        extents.append(chunk[:, 0].max() - chunk[:, 0].min())
    # Larger radii should yield wider extents
    assert extents[0] < extents[1] < extents[2]

    # Scalar (backward compat)
    cyl_actor_scalar = actor.cylinder(
        centers=centers, colors=colors, radii=0.5, height=1.0
    )
    assert cyl_actor_scalar.prim_count == 3

    # Wrong-size array raises ValueError
    with pytest.raises(ValueError, match="radii"):
        actor.cylinder(
            centers=centers,
            colors=colors,
            radii=np.array([0.3, 0.5]),
        )

    with pytest.raises(ValueError, match="height"):
        actor.cylinder(
            centers=centers,
            colors=colors,
            height=np.array([1.0, 1.5]),
        )


def test_actor_from_primitive_wireframe():
    """Test wireframe and wireframe_thickness for primitive actors."""
    sphere_actor = actor.sphere(
        centers=np.array([[0, 0, 0]]), colors=np.array([[1, 0, 0]])
    )

    # By default, wireframe is off
    assert not sphere_actor.material.wireframe
    assert sphere_actor.material.wireframe_thickness == 1.0

    # Test enabling wireframe
    sphere_actor.material.wireframe = True
    assert sphere_actor.material.wireframe

    # Test disabling wireframe
    sphere_actor.material.wireframe = False
    assert not sphere_actor.material.wireframe

    # Test setting wireframe thickness
    new_thickness = 5.0
    sphere_actor.material.wireframe_thickness = new_thickness
    assert sphere_actor.material.wireframe_thickness == new_thickness


def test_cylinder_accepts_255_colors():
    """Cylinder actor accepts [0, 255] colors."""
    centers = np.array([[0, 0, 0]])
    a1 = actor.cylinder(centers=centers, colors=(255, 0, 0))
    a2 = actor.cylinder(centers=centers, colors=(1.0, 0.0, 0.0))
    np.testing.assert_array_almost_equal(
        a1.geometry.colors.view, a2.geometry.colors.view
    )


def test_sphere_accepts_hex_colors():
    """Sphere actor (non-impostor) accepts hex color strings."""
    centers = np.array([[0, 0, 0]])
    a1 = actor.sphere(centers=centers, colors="#FF0000", impostor=False)
    a2 = actor.sphere(centers=centers, colors=(1.0, 0.0, 0.0), impostor=False)
    np.testing.assert_array_almost_equal(
        a1.geometry.colors.view, a2.geometry.colors.view
    )


@pytest.mark.parametrize("impostor", [False, True], ids=["mesh", "impostor"])
@pytest.mark.parametrize("color", ["#FF0000", (1, 0, 0), (255, 0, 0)])
def test_ellipsoid_accepts_hex_colors(impostor, color):
    obj = actor.ellipsoid(
        [[0, 0, 0]], lengths=(2, 1, 0.5), colors=color, impostor=impostor
    )
    with render_manager(obj) as show_m:
        rgb = render_snapshot(show_m)[128, 128, :3]
        assert rgb[0] > 2 * max(int(rgb[1]), int(rgb[2]))
        assert rgb[0] > 20


CURVED_PBR_ACTORS = ["sphere", "ellipsoid", "cylinder", "cone"]

# PBR material tests exercise the explicitly requested mesh branch.
CURVED_PBR_BUILD_KWARGS = {
    "sphere": {"impostor": False},
    "ellipsoid": {"impostor": False},
}


@pytest.mark.parametrize("actor_name", CURVED_PBR_ACTORS)
@pytest.mark.parametrize("mesh_material", ["standard", "physical"])
def test_curved_actors_support_pbr(actor_name, mesh_material):
    assert_supports_pbr(
        actor_name, mesh_material, **CURVED_PBR_BUILD_KWARGS.get(actor_name, {})
    )


@pytest.mark.parametrize("actor_name", CURVED_PBR_ACTORS)
def test_curved_actors_reject_pbr_params_on_phong(actor_name):
    assert_rejects_pbr_params_on_phong(
        actor_name, **CURVED_PBR_BUILD_KWARGS.get(actor_name, {})
    )


def test_sphere_impostor_falls_back_for_pbr_material(caplog):
    # The impostor shader has no PBR path, so a PBR request must switch to real
    # geometry rather than silently dropping the material -- and must say so.
    # This is the one case that leaves impostor at its default; everywhere else
    # asks for geometry up front so no warning is raised.
    with caplog.at_level(logging.WARNING):
        obj = actor.sphere(CENTERS, impostor=True, material="standard")

    assert type(obj.material) is PBR_MATERIAL_TYPES["standard"]
    assert "impostor spheres do not support PBR materials" in caplog.text
    assert "falling back to impostor=False" in caplog.text


@pytest.mark.parametrize("mesh_material", ["standard", "physical"])
def test_streamtube_rejects_pbr_material(mesh_material):
    # Streamtubes render through a baked phong shader, so a PBR request must
    # fail loudly rather than be silently ignored.
    with pytest.raises(ValueError, match="phong"):
        actor.streamtube(LINES, material=mesh_material)


def test_ellipsoid_impostor_surface():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    masks = []
    for axes in (np.eye(3), rotation, np.diag([-1, 1, 1])):
        obj = actor.ellipsoid(
            [[0, 0, 0]],
            lengths=(2, 1, 0.5),
            orientation_matrices=axes,
            colors=(1, 1, 1),
        )
        np.testing.assert_allclose(
            obj.get_bounding_box(),
            [[-1, -2, -0.5], [1, 2, 0.5]]
            if np.array_equal(axes, rotation)
            else [[-2, -1, -0.5], [2, 1, 0.5]],
        )
        with render_manager(obj) as show_m:
            masks.append(snapshot_mask(render_snapshot(show_m)))
    horizontal = snapshot_extents(masks[0])
    vertical = snapshot_extents(masks[1])
    assert abs(horizontal[0] - 2 * horizontal[1]) <= 2
    np.testing.assert_allclose(vertical, horizontal[::-1], atol=2)
    np.testing.assert_array_equal(masks[0], masks[2])


@pytest.mark.parametrize(
    "case",
    ["orthographic", "oblique", "shear", "mirror", "closeup", "near-plane"],
)
def test_ellipsoid_impostor_mesh_silhouette(case):
    world = np.eye(4)
    camera_kwargs = {}
    lengths = (2, 1, 0.5)
    if case == "oblique":
        camera_kwargs = {"perspective": True, "position": (5, 3, 8)}
    elif case == "shear":
        world[:3, :3] = [[1.2, 0.4, 0.2], [0, 0.7, 0.3], [0, 0, 1.5]]
        camera_kwargs = {"perspective": True, "position": (4, 2, 9)}
    elif case == "mirror":
        world[0, 0] = -1
    elif case == "closeup":
        camera_kwargs = {
            "perspective": True,
            "position": (2.5, 0.5, 2),
            "target": (0.7, 0, 0),
        }
    elif case == "near-plane":
        lengths = (1.2, 0.8, 1.35)
        camera_kwargs = {
            "perspective": True,
            "position": (0, 0, 3),
            "near": 1.8,
        }
    masks = []
    for impostor in (False, True):
        obj = actor.ellipsoid(
            [[0, 0, 0]],
            lengths=lengths,
            colors=(1, 1, 1),
            impostor=impostor,
            phi=64,
            theta=64,
        )
        obj.local.state_basis = "matrix"
        obj.local.matrix = world
        if case == "mirror":
            obj.material.side = "both"
        if impostor:
            np.testing.assert_allclose(
                obj.get_bounding_box(), [-np.asarray(lengths), np.asarray(lengths)]
            )
            extent = np.linalg.norm(world[:3, :3] * lengths, axis=1)
            if case != "shear":
                np.testing.assert_allclose(
                    obj.get_world_bounding_box(), [-extent, extent]
                )
        with render_manager(obj, **camera_kwargs) as show_m:
            masks.append(snapshot_mask(render_snapshot(show_m)))
    union = np.count_nonzero(masks[0] | masks[1])
    assert union > 0
    assert np.count_nonzero(masks[0] & masks[1]) / union >= 0.95


def test_ellipsoid_impostor_depth_pick():
    # Glyph 1 has the nearer center, but glyph 0 has the nearer curved surface.
    glyphs = actor.ellipsoid(
        [[0, 0, 0], [0, 0, 0.5]],
        lengths=[[2, 1, 2], [2, 1, 0.5]],
        colors=[[1, 0, 0], [0, 1, 0]],
    )
    mesh = actor.ellipsoid(
        [[0, 0, 0]],
        lengths=(1.8, 0.8, 0.5),
        colors=(0, 0, 1),
        impostor=False,
        phi=64,
        theta=64,
    )
    with render_manager(glyphs, mesh) as show_m:
        image = render_snapshot(show_m)
        for point in ((128, 128), (160, 128)):
            pick = show_m.renderer.get_pick_info(point)
            assert pick["world_object"] is glyphs
            assert pick["glyph_index"] == 0
            rgb = image[point[1], point[0], :3]
            assert rgb[0] > max(rgb[1], rgb[2])
        assert show_m.renderer.get_pick_info((188, 156))["world_object"] is None
        mesh.local.position = (0, 0, 2)
        image = render_snapshot(show_m)
        assert show_m.renderer.get_pick_info((128, 128))["world_object"] is mesh
        assert image[128, 128, 2] > image[128, 128, 0]
        mesh.visible = False
        glyphs.visible = False
        assert not snapshot_mask(render_snapshot(show_m)).any()
        assert show_m.renderer.get_pick_info((128, 128))["world_object"] is None


@pytest.mark.parametrize("options", [{}, {"impostor": True}, {"impostor": False}])
def test_ellipsoid_behavior_warning(caplog, options):
    with caplog.at_level(logging.WARNING, logger="fury.actor.curved"):
        obj = actor.ellipsoid(
            [[0, 0, 0]], lengths=(2, 1, 0.5), phi=6, theta=6, **options
        )
    records = [
        record for record in caplog.records if record.name == "fury.actor.curved"
    ]
    impostor = options.get("impostor", True)
    assert len(records) == int(impostor)
    if impostor:
        assert records[0].levelno == logging.WARNING
    with render_manager(obj) as show_m:
        image = render_snapshot(show_m)
        assert image[128, 128, 0] > max(image[128, 128, 1:3])
        pick = show_m.renderer.get_pick_info((128, 128))
        assert pick["world_object"] is obj
        if impostor:
            assert pick["glyph_index"] == 0
            assert "face_index" not in pick
        else:
            assert "face_index" in pick
            assert "glyph_index" not in pick


@pytest.mark.parametrize(
    "options",
    [
        {"material": "basic"},
        {"material": "standard", "material_params": {"roughness": 0.2}},
        {"wireframe": True, "wireframe_thickness": 2},
        {"smooth": False},
        {"material_params": {"opacity": 0.5}},
    ],
)
def test_ellipsoid_material_fallback(caplog, options):
    with caplog.at_level(logging.WARNING):
        obj = actor.ellipsoid([[0, 0, 0]], lengths=(2, 1, 0.5), **options)
    records = [
        record for record in caplog.records if record.name == "fury.actor.curved"
    ]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    if options.get("material") == "standard":
        assert type(obj.material) is PBR_MATERIAL_TYPES["standard"]
        assert obj.material.roughness == pytest.approx(0.2)
    if options.get("wireframe"):
        assert obj.material.wireframe
        assert obj.material.wireframe_thickness == 2
    if options.get("smooth") is False:
        assert obj.material.flat_shading
    if "opacity" in options.get("material_params", {}):
        assert obj.material.opacity == 0.5
    with render_manager(obj) as show_m:
        image = render_snapshot(show_m)
        assert snapshot_mask(image).any()
        if options.get("material") == "basic":
            np.testing.assert_array_equal(image[128, 128, :3], [255, 0, 0])
    reference = actor.ellipsoid(
        [[0, 0, 0]], lengths=(2, 1, 0.5), impostor=False, **options
    )
    with render_manager(reference) as show_m:
        np.testing.assert_array_equal(image, render_snapshot(show_m))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"centers": [[0, 0]]},
        {"centers": [[np.nan, 0, 0]]},
        {"lengths": (1, 2)},
        {"lengths": (-1, 1, 1)},
        {"lengths": (1, np.inf, 1)},
        {"lengths": np.ones((3, 3))},
        {"orientation_matrices": np.eye(2)},
        {"orientation_matrices": np.ones((3, 3, 3))},
        {"orientation_matrices": np.full((3, 3), np.nan)},
        {"colors": (1, np.nan, 0)},
        {"colors": np.ones((3, 3))},
        {"centers": [[np.finfo(np.float64).max, 0, 0]]},
        {"lengths": (np.finfo(np.float64).max, 1, 1)},
    ],
)
def test_ellipsoid_impostor_invalid_inputs(kwargs):
    params = {"centers": [[0, 0, 0], [2, 0, 0]]}
    params.update(kwargs)
    with pytest.raises(ValueError):
        actor.ellipsoid(**params)


@pytest.mark.parametrize("empty", [False, True], ids=["zero-axis", "empty"])
def test_ellipsoid_impostor_empty_surface(empty):
    obj = actor.ellipsoid(
        np.empty((0, 3)) if empty else [[0, 0, 0]], lengths=(2, 0, 0.5)
    )
    if empty:
        assert obj.get_bounding_box() is None
        assert obj.glyph_count == 0
    with render_manager(obj) as show_m:
        assert not snapshot_mask(render_snapshot(show_m)).any()
        assert show_m.renderer.get_pick_info((128, 128))["world_object"] is None


@pytest.mark.parametrize("impostor", [False, True], ids=["mesh", "impostor"])
def test_ellipsoid_rgba_opacity_and_transforms(impostor):
    obj = actor.ellipsoid(
        [[0, 0, 0]],
        lengths=(2, 1, 0.5),
        colors=(1, 0, 0, 0.5),
        impostor=impostor,
    )
    with render_manager(obj) as show_m:
        initial = render_snapshot(show_m)
        assert initial[128, 128, 0] > initial[128, 128, 1]
        obj.opacity = 0.3
        faded = render_snapshot(show_m)
        assert 0 < faded[128, 128, 0] < initial[128, 128, 0]
        obj.opacity = 1
        world = np.eye(4)
        world[:3, :3] = [[0, -1, 0], [0.5, 0, 0], [0, 0, 1]]
        world[0, 3] = 1
        obj.local.matrix = world
        transformed = render_snapshot(show_m)
        before = snapshot_extents(snapshot_mask(initial))
        after = snapshot_extents(snapshot_mask(transformed))
        np.testing.assert_allclose(after, [before[1], before[0] / 2], atol=2)
        y, x = np.nonzero(snapshot_mask(transformed))
        assert abs(x.mean() - 159.5) <= 2
        assert abs(y.mean() - 127.5) <= 2
        obj.visible = False
        assert not snapshot_mask(render_snapshot(show_m)).any()


def test_ellipsoid_impostor_single_center_and_disabled_pick():
    obj = actor.ellipsoid(
        (0, 0, 0),
        orientation_matrices=np.eye(3)[None],
        lengths=np.array([[2, 1, 0.5]]),
        enable_picking=False,
    )
    with render_manager(obj) as show_m:
        assert snapshot_mask(render_snapshot(show_m)).any()
        assert show_m.renderer.get_pick_info((128, 128))["world_object"] is None


def test_ellipsoid_impostor_singular_transform():
    obj = actor.ellipsoid([[0, 0, 0]], lengths=(2, 1, 0.5))
    obj.local.scale = (1, 1, 0)
    with render_manager(obj) as show_m:
        assert not snapshot_mask(render_snapshot(show_m)).any()
        assert show_m.renderer.get_pick_info((128, 128))["world_object"] is None
