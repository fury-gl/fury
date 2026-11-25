"""Tests for GPU keyframe animation."""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.interpolate import CubicSpline

from fury import actor, window
from fury.lib import BufferUsage, gfx_wgpu
from fury.motion import Animation

RNG = np.random.default_rng(0)


def _sphere(n):
    return actor.sphere(
        np.zeros((n, 3)),
        colors=RNG.random((n, 3)),
        radii=np.full(n, 0.5),
        impostor=True,
    )


def _show_manager():
    scene = window.Scene()
    return window.ShowManager(
        scene=scene,
        size=(128, 128),
        pixel_ratio=1,
        window_type="offscreen",
    )


def _readable(*buffers):
    for buffer in buffers:
        buffer._wgpu_usage |= BufferUsage.COPY_SRC


def read_buffer(buffer):
    """Read a rendered pygfx buffer back from the GPU (tests only)."""
    raw = gfx_wgpu.get_shared().device.queue.read_buffer(buffer._wgpu_object)
    return np.frombuffer(raw, dtype=np.float32).reshape(buffer.data.shape).copy()


def test_gpu_track_cpu_matches_scipy():
    n, times = 4, [0, 1, 3, 4, 7]
    centers = {t: RNG.random((n, 3)) for t in times}
    radii = {t: RNG.random(n) for t in times}
    animation = Animation(actors=_sphere(n))
    animation.set_gpu_keyframes("centers", centers, interpolation="cubic")
    animation.set_gpu_keyframes("radii", radii)

    stacked = np.stack([centers[t] for t in times])
    spline = CubicSpline(times, stacked, axis=0)
    for t in [0, 0.5, 2.2, 6.9, 7]:
        npt.assert_allclose(animation.get_value("centers", t), spline(t), rtol=1e-5)
    npt.assert_allclose(animation.get_value("centers", -1), stacked[0], rtol=1e-6)
    npt.assert_allclose(animation.get_value("centers", 9), stacked[-1], rtol=1e-6)

    r = np.stack([radii[t] for t in times])
    npt.assert_allclose(
        animation.get_value("radii", 2.2), r[1] + 0.6 * (r[2] - r[1]), rtol=1e-5
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_gpu_keyframes_written_by_gpu(interpolation):
    n = 5
    sphere = _sphere(n)
    _readable(sphere.geometry.positions, sphere.geometry.normals)
    times = [0, 1, 3, 4]
    animation = Animation(actors=sphere)
    animation.set_gpu_keyframes(
        "centers",
        {t: RNG.random((n, 3)) * 2 - 1 for t in times},
        interpolation=interpolation,
    )
    animation.set_gpu_keyframes("radii", {t: RNG.random(n) * 0.5 + 0.1 for t in times})
    show_m = _show_manager()
    show_m.add_animation(animation)

    animation.update_animation(time=2.5)
    show_m._draw_function()

    expected_centers = np.repeat(animation.get_value("centers", 2.5), 6, axis=0)
    npt.assert_allclose(
        read_buffer(sphere.geometry.positions),
        expected_centers,
        rtol=1e-5,
        atol=1e-4,
    )
    diameters = np.repeat(2 * animation.get_value("radii", 2.5), 6)
    npt.assert_allclose(
        read_buffer(sphere.geometry.normals)[:, :2],
        np.stack([diameters, diameters], axis=1),
        rtol=1e-5,
        atol=1e-4,
    )


def test_gpu_keyframes_lazy_bbox():
    n = 6
    sphere = _sphere(n)
    animation = Animation(actors=sphere)
    animation.set_gpu_keyframes(
        "centers", {0: RNG.random((n, 3)), 5: RNG.random((n, 3)) * 10}
    )
    animation.set_gpu_keyframes("radii", {0: RNG.random(n), 5: RNG.random(n)})

    animation.update_animation(time=3)
    aabb = sphere.get_bounding_box()

    centers = animation.get_value("centers", 3)
    radii = animation.get_value("radii", 3)
    npt.assert_allclose(sphere.billboard_centers, centers, rtol=1e-5)
    npt.assert_allclose(sphere.geometry.positions.data[::6], centers, rtol=1e-5)
    npt.assert_allclose(aabb[0], (centers - radii[:, None]).min(axis=0), rtol=1e-4)
    npt.assert_allclose(aabb[1], (centers + radii[:, None]).max(axis=0), rtol=1e-4)


def test_gpu_keyframes_vec4_and_raw_buffer():
    n = 3
    centers = RNG.random((n, 3))
    billboards = actor.billboard(centers, colors=RNG.random((n, 4)))
    colors = billboards.geometry.colors
    assert colors.data.shape == (6 * n, 4)
    box = actor.box(np.zeros((1, 3)))
    positions = box.geometry.positions
    assert positions.data.shape == (24, 3)
    original = positions.data.copy()
    _readable(colors, positions)

    c0, c1 = RNG.random((n, 4)), RNG.random((n, 4))
    y0, y1 = RNG.random(24), RNG.random(24)
    anim_colors = Animation(actors=billboards)
    anim_colors.set_gpu_keyframes("colors", {0: c0, 2: c1})
    anim_box = Animation(actors=box)
    anim_box.set_gpu_keyframes("positions", {0: y0, 2: y1}, repeat=1, channels=(1,))

    show_m = _show_manager()
    show_m.add_animation(anim_colors)
    show_m.add_animation(anim_box)
    anim_colors.update_animation(time=1)
    anim_box.update_animation(time=1)
    show_m._draw_function()

    npt.assert_allclose(
        read_buffer(colors),
        np.repeat((c0 + c1) / 2, 6, axis=0),
        rtol=1e-5,
        atol=1e-5,
    )
    out = read_buffer(positions)
    npt.assert_allclose(out[:, 1], (y0 + y1) / 2, rtol=1e-5, atol=1e-5)
    npt.assert_allclose(out[:, [0, 2]], original[:, [0, 2]], rtol=1e-6)


def test_gpu_keyframes_errors():
    n = 3
    value = RNG.random((n, 3))

    def new():
        return Animation(actors=_sphere(n))

    with pytest.raises(ValueError, match="interpolation must be"):
        new().set_gpu_keyframes("centers", {0: value}, interpolation="step")
    with pytest.raises(ValueError, match="must not be empty"):
        new().set_gpu_keyframes("centers", {})
    with pytest.raises(ValueError, match="same shape"):
        new().set_gpu_keyframes("centers", {0: value, 1: RNG.random((n + 1, 3))})
    with pytest.raises(ValueError, match="expected 4 elements x repeat 6"):
        new().set_gpu_keyframes("centers", {0: RNG.random((n + 1, 3))})
    with pytest.raises(ValueError, match="predefined GPU attribute"):
        new().set_gpu_keyframes("centers", {0: value}, repeat=2)
    with pytest.raises(ValueError, match="neither a GPU attribute"):
        new().set_gpu_keyframes("foo", {0: value})

    animation = new()
    animation.set_gpu_keyframes("radii", {0: RNG.random(n)})
    with pytest.raises(ValueError, match="same buffer channels"):
        animation.set_gpu_keyframes("sizes", {0: RNG.random((n, 2))})

    with pytest.raises(ValueError, match="actor must be given"):
        Animation(actors=[_sphere(n), _sphere(n)]).set_gpu_keyframes(
            "centers", {0: value}
        )

    animation = new()
    animation.set_keyframe("centers", 0, value)
    with pytest.raises(ValueError, match="already has CPU keyframes"):
        animation.set_gpu_keyframes("centers", {0: value})
