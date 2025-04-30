from numpy import ceil, prod

from fury.lib import (
    BaseShader,
    Binding,
    Buffer,
    ThinLineSegmentShader,
    load_wgsl,
)


class VectorFieldComputeShader(BaseShader):
    """Compute shader for vector field."""

    type = "compute"

    def __init__(self, wobject):
        super().__init__(wobject)
        self["num_vectors"] = wobject.vectors_per_voxel
        self["data_shape"] = wobject.field_shape
        self["workgroup_size"] = 1024

    def get_render_info(self, wobject, _shared):
        n = int(ceil(prod(wobject.field_shape) / self["workgroup_size"]))
        return {
            "indices": (n, 1, 1),
        }

    def get_pipeline_info(self, _wobject, _shared):
        return {}

    def get_bindings(self, wobject, _shared):
        # To share the bindings across compute and render shaders, we need to
        # define the bindings exactly the same way in both shaders.
        bindings = {
            0: Binding(
                "s_vectors", "buffer/storage", Buffer(wobject.vectors), "COMPUTE"
            ),
            1: Binding("s_scales", "buffer/storage", Buffer(wobject.scales), "COMPUTE"),
            3: Binding(
                "s_positions", "buffer/storage", wobject.geometry.positions, "COMPUTE"
            ),
            4: Binding(
                "s_colors", "buffer/storage", wobject.geometry.colors, "COMPUTE"
            ),
        }
        self.define_bindings(0, bindings)

        return {0: bindings}

    def get_code(self):
        return load_wgsl("vector_field_compute.wgsl", package_name="fury.wgsl")


class VectorFieldShader(ThinLineSegmentShader):
    """Shader for VectorFieldActor."""

    def __init__(self, wobject):
        super().__init__(wobject)
        self["num_vectors"] = wobject.vectors_per_voxel
        self["data_shape"] = wobject.field_shape

    def get_code(self):
        return load_wgsl("vector_field_render.wgsl", package_name="fury.wgsl")
