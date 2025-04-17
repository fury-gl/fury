import numpy as np

from fury.geometry import buffer_to_geometry, line_buffer_separator
from fury.lib import (
    LineMaterial,
    LineShader,
    WorldObject,
    register_wgpu_render_function,
)
from fury.material import _create_line_material


class PeaksActor(WorldObject):
    def __init__(
        self,
        directions,
        indices,
        values,
        *,
        colors=None,
        line_width=1,
        symmetric=True,
    ):
        valid_dirs = directions[indices]
        # num_dirs = len(np.nonzero(np.abs(valid_dirs).max(axis=-1) > 0)[0])

        # Two points and a np.nan buffer to separate lines
        # pnts_per_line = 3

        # points = np.empty((num_dirs * pnts_per_line, 3), dtype=np.float32)
        lines = []
        # centers = np.empty_like(points, dtype=np.int32)
        # diffs = np.empty_like(points)
        line_count = 0

        for idx, center in enumerate(zip(indices[0], indices[1], indices[2])):
            xyz = np.asarray(center, dtype=np.int32)
            # print(center.shape)
            valid_peaks = np.nonzero(np.abs(valid_dirs[idx, :, :]).max(axis=-1) > 0.0)[
                0
            ]

            for direction in valid_peaks:
                # p_value = values[center][direction]
                p_value = directions[center][direction] * 0.5

                point_i = p_value + xyz
                point_e = -1 * p_value + xyz if symmetric else xyz

                # diff = point_e - point_i

                # points[line_count * pnts_per_line] = point_i
                # points[line_count * pnts_per_line + 1] = point_e
                # points[line_count * pnts_per_line + 2] = point_nan

                lines.append([point_i, point_e])
                # centers[line_count * pnts_per_line] = center
                # centers[line_count * pnts_per_line + 1] = center
                # centers[line_count * pnts_per_line + 2] = center
                # diffs[line_count * pnts_per_line] = diff
                # diffs[line_count * pnts_per_line + 1] = diff
                # diffs[line_count * pnts_per_line + 2] = diff
                line_count += 1

        # if colors is None:
        #     num_points = points.shape[0]
        #     # colors = np.asarray((1, 1, 1), dtype=np.float32)
        #     # colors = np.tile(255 * colors, (num_points, 1))
        #     colors = np.random.rand(num_points, 3).astype(np.float32)
        # else:
        #     colors = colors.astype(np.float32)
        colors = np.random.rand(line_count, 3).astype(np.float32)
        points, colors = line_buffer_separator(lines, color=colors)
        geometry = buffer_to_geometry(positions=points, colors=colors)
        material = _create_line_material(thickness=line_width)
        super().__init__(geometry=geometry, material=material)


register_wgpu_render_function(PeaksActor, LineMaterial)(LineShader)
# @register_wgpu_render_function(PeaksActor, LineMaterial)
# class PeaksShader(BaseShader):
#     """Shader for PeaksActor."""

#     def __init__(self, wobject):
#         super().__init__(wobject)
