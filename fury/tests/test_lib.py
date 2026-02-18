import numpy as np
import pytest

from fury import lib


class _FakeQueue:
    def __init__(self, raw_bytes):
        self._raw_bytes = raw_bytes

    def read_buffer(self, _buffer):
        return self._raw_bytes


class _FakeShared:
    def __init__(self, raw_bytes):
        self.device = type("Device", (), {"queue": _FakeQueue(raw_bytes)})()


def test_read_buffer_syncs_cpu_data(monkeypatch):
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    buffer = lib.Buffer(np.zeros_like(expected))
    monkeypatch.setattr(
        lib.gfx.renderers.wgpu,
        "get_shared",
        lambda: _FakeShared(expected.tobytes()),
    )

    out = lib.read_buffer(buffer)

    assert np.array_equal(out, expected)
    assert np.array_equal(buffer.data, expected)


def test_read_buffer_without_sync_keeps_cpu_data(monkeypatch):
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    buffer = lib.Buffer(np.zeros_like(expected))
    monkeypatch.setattr(
        lib.gfx.renderers.wgpu,
        "get_shared",
        lambda: _FakeShared(expected.tobytes()),
    )

    out = lib.read_buffer(buffer, sync_cpu=False)

    assert np.array_equal(out, expected)
    assert np.array_equal(buffer.data, np.zeros_like(expected))


def test_read_buffer_rejects_non_buffer_instance():
    with pytest.raises(ValueError, match="Expected a wgpu.Buffer instance."):
        lib.read_buffer(np.zeros((2, 3), dtype=np.float32))
