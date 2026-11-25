"""GPU keyframe animation of actor buffers."""

import numpy as np
from scipy.interpolate import CubicSpline

from fury.lib import Buffer, WorldObject, gfx_wgpu, register_wgpu_render_function
from fury.material import GPUKeyframeMaterial
from fury.shader import GPUKeyframeComputeShader


class _GPUKeyframeTrack:
    """
    CPU-side state of one GPU-animated buffer region.

    Parameters
    ----------
    actor : WorldObject
        The actor owning the target buffer.
    name : str
        Either a predefined GPU attribute of the actor or a geometry buffer
        name.
    keyframes : dict
        Mapping of timestamp to an ``(N,)`` or ``(N, C)`` array of values.
    interpolation : {"linear", "cubic"}
        Interpolation mode.
    repeat : int or None
        Number of buffer rows written per element.
    channels : sequence of int or None
        Buffer columns written.
    scale : float or None
        Multiplier applied to the values when written.
    """

    def __init__(
        self, actor, name, keyframes, *, interpolation, repeat, channels, scale
    ):
        """
        Initialize the track and attach its compute animator to the actor.

        Parameters
        ----------
        actor : WorldObject
            The actor owning the target buffer.
        name : str
            Either a predefined GPU attribute of the actor or a geometry buffer
            name.
        keyframes : dict
            Mapping of timestamp to an ``(N,)`` or ``(N, C)`` array of values.
        interpolation : {"linear", "cubic"}
            Interpolation mode.
        repeat : int or None
            Number of buffer rows written per element.
        channels : sequence of int or None
            Buffer columns written.
        scale : float or None
            Multiplier applied to the values when written.
        """
        if interpolation not in ("linear", "cubic"):
            raise ValueError(
                f"interpolation must be 'linear' or 'cubic', got {interpolation!r}"
            )

        aliases = getattr(actor, "gpu_attributes", {})
        if name in aliases:
            if repeat is not None or channels is not None or scale is not None:
                raise ValueError(
                    f"{name!r} is a predefined GPU attribute of "
                    f"{type(actor).__name__}; repeat, channels and scale must "
                    "not be given"
                )
            spec = aliases[name]
            buffer_name = spec["buffer"]
            repeat = spec["repeat"]
            channels = spec["channels"]
            scale = spec["scale"]
            cpu_attribute = spec["cpu_attribute"]
        elif isinstance(getattr(actor.geometry, name, None), Buffer):
            buffer_name = name
            cpu_attribute = None
            repeat = 1 if repeat is None else int(repeat)
            scale = 1.0 if scale is None else float(scale)
        else:
            raise ValueError(
                f"{name!r} is neither a GPU attribute ({sorted(aliases)}) nor a "
                f"geometry buffer of {type(actor).__name__}"
            )

        self.buffer = getattr(actor.geometry, buffer_name)
        data = self.buffer.data
        if data.dtype != np.float32 or data.ndim not in (1, 2):
            raise ValueError(
                f"Buffer {buffer_name!r} must be a 1-D or 2-D float32 array"
            )
        self.stride = 1 if data.ndim == 1 else data.shape[1]
        channels = (
            tuple(range(self.stride))
            if channels is None
            else tuple(int(c) for c in channels)
        )
        if (
            not channels
            or len(set(channels)) != len(channels)
            or any(not 0 <= c < self.stride for c in channels)
        ):
            raise ValueError(
                f"channels {channels} are invalid for buffer {buffer_name!r} "
                f"with {self.stride} columns"
            )

        if not keyframes:
            raise ValueError("keyframes must not be empty")
        items = sorted(keyframes.items(), key=lambda kv: float(kv[0]))
        arrays = [np.asarray(v, dtype=np.float32) for _, v in items]
        first = arrays[0]
        if first.ndim not in (1, 2) or any(a.shape != first.shape for a in arrays):
            raise ValueError(
                "all keyframe values must have the same shape (N,) or (N, C)"
            )
        self.value_shape = first.shape
        self.times = np.array([float(t) for t, _ in items], dtype=np.float32)
        self.values = np.stack(arrays)
        if self.values.ndim == 2:
            self.values = self.values[:, :, None]
        n_keys, self.n_elements, n_comp = self.values.shape
        if n_comp not in (1, len(channels)):
            raise ValueError(
                f"keyframe values have {n_comp} components; expected 1 or "
                f"{len(channels)} for channels {channels}"
            )
        if data.shape[0] != self.n_elements * repeat:
            raise ValueError(
                f"buffer {buffer_name!r} has {data.shape[0]} rows; expected "
                f"{self.n_elements} elements x repeat {repeat}"
            )

        times64 = self.times.astype(np.float64)
        values64 = self.values.astype(np.float64)
        if interpolation == "cubic":
            if n_keys >= 2:
                self.slopes = (
                    CubicSpline(times64, values64, axis=0)
                    .derivative()(times64)
                    .astype(np.float32)
                )
            else:
                self.slopes = np.zeros_like(self.values)
        else:
            self.slopes = None

        limit = gfx_wgpu.get_shared().device.limits["max-storage-buffer-binding-size"]
        if self.values.nbytes > limit:
            raise ValueError(
                f"GPU keyframes for {name!r} need {self.values.nbytes} bytes, "
                f"above the device storage binding limit of {limit} bytes"
            )

        self.actor = actor
        self.name = name
        self.interpolation = interpolation
        self.repeat = repeat
        self.channels = channels
        self.scale = scale
        self.cpu_attribute = cpu_attribute
        self.time = 0.0
        self._synced_time = None
        self.animator = _GPUKeyframeAnimator(self)
        actor.add(self.animator)
        self.sync_to_cpu()

    def evaluate(self, timestamp):
        """
        Evaluate the keyframe curve on the CPU, mirroring the GPU kernel.

        Parameters
        ----------
        timestamp : float
            Time to evaluate at. Clamped to the first and last keyframe.

        Returns
        -------
        ndarray
            Float64 values shaped like a keyframe value, without scale applied.
        """
        times = self.times
        n_keys = times.shape[0]
        t = float(np.clip(timestamp, times[0], times[-1]))
        if n_keys == 1:
            out = self.values[0].astype(np.float64)
        else:
            k = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, n_keys - 2))
            h = float(times[k + 1]) - float(times[k])
            s = (t - float(times[k])) / h if h > 0 else 0.0
            v0 = self.values[k].astype(np.float64)
            v1 = self.values[k + 1].astype(np.float64)
            if self.slopes is None:
                out = (1 - s) * v0 + s * v1
            else:
                m0 = self.slopes[k].astype(np.float64)
                m1 = self.slopes[k + 1].astype(np.float64)
                s2 = s * s
                s3 = s2 * s
                out = (
                    (2 * s3 - 3 * s2 + 1) * v0
                    + (s3 - 2 * s2 + s) * h * m0
                    + (-2 * s3 + 3 * s2) * v1
                    + (s3 - s2) * h * m1
                )
        return out.reshape(self.value_shape)

    def set_time(self, timestamp):
        """
        Set the animation time sent to the GPU.

        Parameters
        ----------
        timestamp : float
            The new animation time.
        """
        if float(timestamp) != self.time:
            self.time = float(timestamp)
            self.animator.material.time = self.time

    def sync_to_cpu(self):
        """Write the currently drawn values into the CPU copy of the buffer."""
        if self._synced_time == self.time:
            return
        n = self.n_elements
        v = self.evaluate(self.time).reshape(n, -1)
        v = np.broadcast_to(v, (n, len(self.channels))) * self.scale
        view = self.buffer.data.reshape(n, self.repeat, self.stride)
        view[:, :, list(self.channels)] = v[:, None, :]
        self.buffer.update_full()
        if self.cpu_attribute is not None:
            setattr(self.actor, self.cpu_attribute, v)
        self._synced_time = self.time

    def detach(self):
        """Remove the compute animator from the actor."""
        self.actor.remove(self.animator)


class _GPUKeyframeAnimator(WorldObject):
    """
    Invisible compute-only child object driving a GPU keyframe track.

    Parameters
    ----------
    track : _GPUKeyframeTrack
        The track to animate.
    """

    def __init__(self, track):
        """
        Initialize the animator and upload keyframes to GPU buffers.

        Parameters
        ----------
        track : _GPUKeyframeTrack
            The track to animate.
        """
        super().__init__(material=GPUKeyframeMaterial(scale=track.scale))
        self.track = track
        self.times_buffer = Buffer(track.times)
        self.values_buffer = Buffer(track.values.reshape(-1))
        self.slopes_buffer = (
            None if track.slopes is None else Buffer(track.slopes.reshape(-1))
        )

    def get_bounding_box(self):
        """
        Sync the CPU copy of the target buffer, then report no bounds.

        Parents query children first, so this is the lazy sync hook.

        Returns
        -------
        None
            The animator has no spatial extent.
        """
        self.track.sync_to_cpu()
        return None


@register_wgpu_render_function(_GPUKeyframeAnimator, GPUKeyframeMaterial)
def _register_gpu_keyframe_shaders(wobject):
    """
    Register the compute shader of a GPU keyframe animator.

    Parameters
    ----------
    wobject : _GPUKeyframeAnimator
        The animator object.

    Returns
    -------
    tuple
        Shader instances for the object.
    """
    return (GPUKeyframeComputeShader(wobject),)
