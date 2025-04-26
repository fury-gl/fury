import numpy as np

from fury.geometry import buffer_to_geometry
from fury.lib import (
    LineThinMaterial,
    WorldObject,
    register_wgpu_render_function,
)
from fury.material import _create_line_material
from fury.shaders.base import PeaksComputeShader, PeaksShader


class PeaksActor(WorldObject):
    def __init__(
        self,
        directions,
        indices,
        values,
        *,
        colors=None,
        symmetric=True,
    ):
        total_vectors = np.prod(directions.shape[:4])
        self.directions = directions.reshape(total_vectors, 3).astype(np.float32)
        print(self.directions)
        pnts_per_line = 2
        self.data_shape = directions.shape[:3]
        self.num_vectors = directions.shape[3]
        self.cross_section = (
            self.data_shape[0] // 2,
            self.data_shape[1] // 2,
            self.data_shape[2] // 2,
        )

        points = np.zeros((total_vectors * pnts_per_line, 3), dtype=np.float32)
        self.centers = np.indices(self.data_shape).reshape(3, -1).T.astype(np.int32)
        # line_count = 0

        if colors is None:
            colors = np.asarray((1, 0, 0), dtype=np.float32)
        colors = np.tile(colors, (total_vectors * pnts_per_line, 1))
        geometry = buffer_to_geometry(positions=points, colors=colors)
        material = _create_line_material(material="thin", mode="vertex")
        # data_shape = directions.shape[:3]
        # self.cross_section = (
        #     data_shape[0] // 2,
        #     data_shape[1] // 2,
        #     data_shape[2] // 2,
        # )

        # self.low_range = (0, 0, 0)
        # self.high_range = data_shape
        print("points_shape", points.shape)
        super().__init__(geometry=geometry, material=material)


@register_wgpu_render_function(PeaksActor, LineThinMaterial)
def register_peaks_shaders(wobject):
    """Register PeaksActor shaders."""
    return PeaksComputeShader(wobject), PeaksShader(wobject)


# @register_wgpu_render_function(PeaksActor, LineMaterial)
# class PeaksShader(BaseShader):
#     """Shader for PeaksActor."""

#     def __init__(self, wobject):
#         super().__init__(wobject)
