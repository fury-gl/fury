from numpy import ceil

from fury.lib import (
    BaseShader,
    Binding,
    Buffer,
    CullMode,
    PrimitiveTopology,
    ThinLineShader,
    load_wgsl,
)


class PeaksComputeShader(BaseShader):
    """Compute shader for peaks detection."""

    type = "compute"

    def __init__(self, wobject):
        super().__init__(wobject)
        self["cross_section"] = wobject.cross_section
        # self["total_vectors"] = (
        #     wobject.data_shape[0] * wobject.data_shape[1]
        #     + wobject.data_shape[1] * wobject.data_shape[2]
        #     + wobject.data_shape[0] * wobject.data_shape[2]
        # )
        self["total_vectors"] = wobject.total_vectors
        self["num_vectors"] = wobject.num_vectors
        self["data_shape"] = wobject.data_shape
        self["workgroup_size"] = 128

    def get_render_info(self, _wobject, _shared):
        n = int(ceil(self["total_vectors"] / self["workgroup_size"]))
        return {
            "indices": (n, 1, 1),
        }

    def get_pipeline_info(self, wobject, shared):
        return {}

    def get_bindings(self, wobject, shared):
        bindings = {
            0: Binding(
                "s_directions", "buffer/storage", Buffer(wobject.directions), "COMPUTE"
            ),
            3: Binding(
                "s_positions", "buffer/storage", wobject.geometry.positions, "COMPUTE"
            ),
            4: Binding(
                "s_colors", "buffer/storage", wobject.geometry.colors, "COMPUTE"
            ),
        }
        self.define_bindings(0, bindings)
        new_bindings = [
            Binding("s_centers", "buffer/storage", Buffer(wobject.centers), "COMPUTE"),
        ]
        new_bindings = dict(enumerate(new_bindings))
        self.define_bindings(1, new_bindings)
        return {0: bindings, 1: new_bindings}

    def get_code(self):
        return load_wgsl("peaks_compute.wgsl", package_name="fury.shaders.wgsl")


class PeaksShader(ThinLineShader):
    """Shader for PeaksActor."""

    def __init__(self, wobject):
        super().__init__(wobject)
        self["cross_section"] = wobject.cross_section
        # self["data_shape"] = wobject.data_shape

    def get_pipeline_info(self, _wobject, _shared):
        return {
            "primitive_topology": PrimitiveTopology.line_list,
            "cull_mode": CullMode.none,
        }

    def get_bindings(self, wobject, shared):
        rbuffer = "buffer/read_only_storage"
        bindings = [
            Binding("s_centers", rbuffer, Buffer(wobject.centers), "VERTEX"),
        ]
        bindings = dict(enumerate(bindings))
        self.define_bindings(1, bindings)
        super_bindings = super().get_bindings(wobject, shared)
        super_bindings[1] = bindings
        return super_bindings

    def get_code(self):
        return load_wgsl("peaks.wgsl", package_name="fury.shaders.wgsl")
