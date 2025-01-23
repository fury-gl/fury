from pygfx.renderers.wgpu.shaders.meshshader import MeshShader

from fury.io import load_wgsl


class MeshBasicShader(MeshShader):
    """Base class for mesh shaders."""

    def __init__(self, wobject):
        super().__init__(wobject)

    def get_code(self):
        return load_wgsl("mesh.wgsl")


class MeshPhongShader(MeshBasicShader):
    """Phong shader for meshes."""

    def __init__(self, wobject):
        super().__init__(wobject)

        self["lighting"] = "phong"
