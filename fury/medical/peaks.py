import numpy as np

from fury.geometry import buffer_to_geometry
from fury.lib import (
    LineThinMaterial,
    WorldObject,
    register_wgpu_render_function,
)
from fury.material import _create_line_material
from fury.shaders.base import PeaksShader


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
        valid_dirs = directions[indices]
        num_dirs = len(np.nonzero(np.abs(valid_dirs).max(axis=-1) > 0)[0])

        pnts_per_line = 3

        points = np.empty((num_dirs * pnts_per_line, 3), dtype=np.float32)
        self.centers = np.empty_like(points, dtype=np.int32)
        self.diffs = np.empty_like(points)
        line_count = 0

        for idx, center in enumerate(zip(indices[0], indices[1], indices[2])):
            xyz = np.asarray(center, dtype=np.int32)
            valid_peaks = np.nonzero(np.abs(valid_dirs[idx, :, :]).max(axis=-1) > 0.0)[
                0
            ]

            for direction in valid_peaks:
                p_value = directions[center][direction] * values[center][direction]

                point_i = p_value + xyz
                point_e = -1 * p_value + xyz if symmetric else xyz
                point_nan = np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)

                diff = point_e - point_i

                points[line_count * pnts_per_line] = point_i
                points[line_count * pnts_per_line + 1] = point_e
                points[line_count * pnts_per_line + 2] = point_nan

                self.centers[line_count * pnts_per_line] = center
                self.centers[line_count * pnts_per_line + 1] = center
                self.centers[line_count * pnts_per_line + 2] = center
                self.diffs[line_count * pnts_per_line] = diff
                self.diffs[line_count * pnts_per_line + 1] = diff
                self.diffs[line_count * pnts_per_line + 2] = diff
                line_count += 1

        num_points = points.shape[0]
        if colors is None:
            colors = np.asarray((0, 0, 0), dtype=np.float32)
        colors = np.tile(colors, (num_points, 1))
        geometry = buffer_to_geometry(positions=points, colors=colors)
        material = _create_line_material(material="thin", mode="vertex")
        data_shape = directions.shape[:3]
        self.cross_section = (
            data_shape[0] // 2,
            data_shape[1] // 2,
            data_shape[2] // 2,
        )

        self.low_range = (0, 0, 0)
        self.high_range = data_shape

        self.is_cross_section = True
        self.is_ranges = False
        super().__init__(geometry=geometry, material=material)

    def show_ranges(self):
        """Show ranges."""
        self.is_ranges = True
        self.is_cross_section = False

    def show_cross_section(self):
        """Show cross section."""
        self.is_cross_section = True
        self.is_ranges = False

    def move_ranges(self, *, x0=None, y0=None, z0=None, x1=None, y1=None, z1=None):
        """Move ranges.

        Parameters
        ----------
        x0 : float, optional
            The lower bound of the range in the x direction.
            If None will assume the current value.
        y0 : float, optional
            The lower bound of the range in the y direction.
            If None will assume the current value.
        z0 : float, optional
            The lower bound of the range in the z direction.
            If None will assume the current value.
        x1 : float, optional
            The upper bound of the range in the x direction.
            If None will assume the current value.
        y1 : float, optional
            The upper bound of the range in the y direction.
            If None will assume the current value.
        z1 : float, optional
            The upper bound of the range in the z direction.
            If None will assume the current value.
        """

        for idx, val in enumerate((x0, y0, z0, x1, y1, z1)):
            if val is None:
                continue
            if idx < 3:
                self.low_range[idx] = val
            else:
                self.high_range[idx - 3] = val
        self.low_range = (x0, y0, z0)
        self.high_range = (x1, y1, z1)


register_wgpu_render_function(PeaksActor, LineThinMaterial)(PeaksShader)
# @register_wgpu_render_function(PeaksActor, LineMaterial)
# class PeaksShader(BaseShader):
#     """Shader for PeaksActor."""

#     def __init__(self, wobject):
#         super().__init__(wobject)
