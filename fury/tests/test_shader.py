import numpy as np
from pygfx import Mesh
from pygfx.renderers.wgpu import register_wgpu_render_function

from fury.actor import sphere
from fury.io import load_wgsl
from fury.material import MeshBasicMaterial
from fury.shader import MeshBasicShader, MeshPhongShader


def test_shader():
    class CustomBasicMaterial(MeshBasicMaterial):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class CustomPhongMaterial(MeshBasicMaterial):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    register_wgpu_render_function(Mesh, CustomBasicMaterial)(MeshBasicShader)
    register_wgpu_render_function(Mesh, CustomPhongMaterial)(MeshPhongShader)

    try:
        register_wgpu_render_function(Mesh, CustomBasicMaterial)(MeshBasicShader)
        register_wgpu_render_function(Mesh, CustomPhongMaterial)(MeshPhongShader)
    except ValueError:
        ...
    else:
        raise AssertionError("Shouldn't be able to register the same material twice.")


def test_wgsl():
    shader_code = load_wgsl("mesh.wgsl")

    assert isinstance(shader_code, str)
    assert "fn vs_main" in shader_code
    assert "fn fs_main" in shader_code

    actor = sphere(centers=np.array([[0, 0, 0]]), colors=np.array([[1, 0, 0]]))
    kwargs = {
        "blending_code": "placeholder",
        "write_pick": True,
        "indexer": None,
        "used_uv": {"uv": None},
    }

    cs = MeshBasicShader(actor)

    assert isinstance(cs.get_code(), str)

    gen_sh_code = cs.generate_wgsl(**kwargs)
    assert isinstance(gen_sh_code, str)

    assert "placeholder" in gen_sh_code
