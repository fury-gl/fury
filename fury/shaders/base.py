from fury.lib import (
    Binding,
    Buffer,
    ThinLineShader,
    load_wgsl,
)


class PeaksShader(ThinLineShader):
    """Shader for PeaksActor."""

    def __init__(self, wobject):
        super().__init__(wobject)
        self["cross_section"] = wobject.cross_section
        self["is_ranges"] = wobject.is_ranges
        self["is_cross_section"] = wobject.is_cross_section
        self["low_range"] = wobject.low_range
        self["high_range"] = wobject.high_range

    def get_bindings(self, wobject, shared):
        rbuffer = "buffer/read_only_storage"
        bindings = [
            Binding("s_centers", rbuffer, Buffer(wobject.centers), "VERTEX"),
            Binding("s_diffs", rbuffer, Buffer(wobject.diffs), "VERTEX"),
        ]
        bindings = dict(enumerate(bindings))
        self.define_bindings(1, bindings)
        super_bindings = super().get_bindings(wobject, shared)
        super_bindings[1] = bindings
        return super_bindings

    def get_code(self):
        return load_wgsl("peaks.wgsl", package_name="fury.shaders.wgsl")
