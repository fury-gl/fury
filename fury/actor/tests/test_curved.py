from PIL import Image
import numpy as np
import pytest

from fury import actor, window
from fury.actor.tests._helpers import validate_actors
from fury.material import (
    StreamlinesMaterial,
    _StreamlineBakedMaterial,
    _StreamtubeBakedMaterial,
)


def test_sphere():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="sphere")


def test_cylinder():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="cylinder")


def test_cone():
    centers = np.array([[0, 0, 0]])
    colors = np.array([[1, 0, 0]])
    validate_actors(centers=centers, colors=colors, actor_type="cone")


def test_ellipsoid():
    centers = np.array([[0, 0, 0]])
    lengths = np.array([[2, 1, 1]])
    axes = np.array([np.eye(3)])
    colors = np.array([1, 0, 0])

    validate_actors(
        centers=centers,
        lengths=lengths,
        orientation_matrices=axes,
        colors=colors,
        actor_type="ellipsoid",
    )

    _ = actor.ellipsoid(
        centers=centers,
        lengths=lengths,
        orientation_matrices=axes,
        colors=colors,
    )

    _ = actor.ellipsoid(
        np.array([[0, 0, 0], [1, 1, 1]]),
        lengths=np.array([[2, 1, 1]]),
        colors=np.array([[1, 0, 0]]),
    )

    _ = actor.ellipsoid(
        np.array([[0, 0, 0], [1, 1, 1]]), lengths=(2, 1, 1), colors=(1, 0, 0)
    )

    _ = actor.ellipsoid(centers)


def test_streamtube():
    lines = [np.array([[0, 0, 0], [1, 1, 1]])]
    colors = np.array([[1, 0, 0]])
    scene = window.Scene()

    tube_actor = actor.streamtube(lines=lines, colors=colors)
    scene.add(tube_actor)

    fname = "streamtube_test.png"
    window.snapshot(scene=scene, fname=fname)
    img = Image.open(fname)
    img_array = np.array(img)

    mean_r, mean_g, mean_b, _ = np.mean(
        img_array.reshape(-1, img_array.shape[2]), axis=0
    )

    assert mean_r > mean_g and mean_r > mean_b

    middle_pixel = img_array[img_array.shape[0] // 2, img_array.shape[1] // 2]
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

    with pytest.raises(
        ValueError, match=r"(must have 3|components, got) \(RGB\) or 4 \(RGBA\)"
    ):
        actor.streamtube(
            [line_a],
            colors=np.array([1.0, 0.5], dtype=np.float32),
            backend="gpu",
        )


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
