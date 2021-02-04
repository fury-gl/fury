import numpy as np
import vtk

from fury.deprecator import deprecate_with_version
from fury.utils import (set_polydata_vertices, set_polydata_triangles,
                        set_polydata_normals, set_polydata_colors)
from fury.colormap import create_colormap


class OdfSlicerActor(vtk.vtkActor):
    def __init__(self, odfs, sphere, indices, scale, norm, radial_scale,
                 global_cm, colormap, opacity, affine=None, mask=None, B=None):
        """
        When B is specified, odfs expressed in SH coeffients
        """
        self.vertices = sphere.vertices
        self.faces = sphere.faces
        self.odfs = odfs
        self.indices = indices
        self.B = B
        self.norm = norm
        self.scale = scale
        self.radial_scale = radial_scale
        self.colormap = colormap
        self.global_cm = global_cm
        if self.B is None:
            if self.norm:
                self.odfs /= np.abs(self.odfs).max(axis=-1, keepdims=True)
            if self.scale:
                self.odfs *= scale

        if mask is not None:
            self.grid_shape = mask.shape
            self.mask = mask
        else:
            self.grid_shape = tuple(np.array(self.indices).max(axis=-1) + 1)
            self.mask = np.ones(self.grid_shape, dtype=np.bool)

        self.w_verts = None
        self.w_pos = None
        if affine is not None:
            self.w_verts = self.vertices.dot(affine[:3, :3])
            self.w_pos =\
                np.append(np.asarray(self.indices).T,
                          np.ones((len(self.odfs), 1)), axis=-1).dot(affine)
            self.w_pos = self.w_pos[..., :-1]

        self.mapper = vtk.vtkPolyDataMapper()
        self.SetMapper(self.mapper)
        self.slice_along_axis(self.grid_shape[-1]//2)
        self.set_opacity(opacity)

    def set_opacity(self, opacity):
        self.GetProperty().SetOpacity(opacity)

    def display_extent(self, x1, x2, y1, y2, z1, z2):
        """
        Set visible volume from x1 (inclusive) to x2 (inclusive),
        y1 (inclusive) to y2 (inclusive), z1 (inclusive) to z2
        (inclusive).
        """
        mask = np.zeros_like(self.mask)
        mask[x1:x2 + 1, y1:y2 + 1, z1:z2 + 1] = True
        mask = np.bitwise_and(mask, self.mask)

        self._update_mapper(mask)

    def slice_along_axis(self, slice_index, axis='zaxis'):
        if axis == 'xaxis':
            self.display_extent(slice_index, slice_index,
                                0, self.grid_shape[1] - 1,
                                0, self.grid_shape[2] - 1)
        elif axis == 'yaxis':
            self.display_extent(0, self.grid_shape[0] - 1,
                                slice_index, slice_index,
                                0, self.grid_shape[2] - 1)
        elif axis == 'zaxis':
            self.display_extent(0, self.grid_shape[0] - 1,
                                0, self.grid_shape[1] - 1,
                                slice_index, slice_index)
        else:
            raise ValueError('Invalid axis name {0}.'.format(axis))

    @deprecate_with_version('Method display() is deprecated. '
                            'Use slice_along_axis() instead.')
    def display(self, x=None, y=None, z=None):
        if x is None and y is None and z is None:
            self.slice_along_axis(self.shape[2]//2)
        elif x is not None:
            self.slice_along_axis(x, 'xaxis')
        elif y is not None:
            self.slice_along_axis(y, 'yaxis')
        elif z is not None:
            self.slice_along_axis(z, 'zaxis')

    def _update_mapper(self, mask):
        # World positions of nonzero voxels
        w_pos = \
            self.w_pos[mask[self.indices]]\
            if self.w_pos is not None\
            else np.asarray(self.indices).T
        sph_dirs = \
            self.w_verts\
            if self.w_verts is not None else\
            self.vertices

        if len(w_pos) == 0:
            return None

        # Convert to SF if input is expressed as SH
        if self.B is not None:
            sf = self.odfs[mask[self.indices]].dot(self.B)
            if self.norm:
                sf /= np.abs(sf).max(axis=-1, keepdims=True)
            sf *= self.scale
        else:
            sf = self.odfs

        if self.radial_scale:
            all_vertices =\
                np.tile(sph_dirs, (len(w_pos), 1)) * sf.reshape(-1, 1) +\
                np.repeat(w_pos, len(sph_dirs), axis=0)
        else:
            all_vertices =\
                np.tile(sph_dirs, (len(w_pos), 1)) * self.scale +\
                np.repeat(w_pos, len(sph_dirs), axis=0)

        all_faces =\
            np.tile(self.faces, (len(w_pos), 1)) + \
            np.repeat(np.arange(len(w_pos)) * len(sph_dirs),
                      len(self.faces)).reshape(-1, 1)

        all_colors = self._generate_color_for_vertices(sf)

        polydata = vtk.vtkPolyData()
        set_polydata_triangles(polydata, all_faces)
        set_polydata_vertices(polydata, all_vertices)
        set_polydata_colors(polydata, all_colors)
        self.mapper.SetInputData(polydata)

    def _generate_color_for_vertices(self, sf):
        """
        *Calls to create_colormap are expensive. To increase
        compute time, use colormap=None.
        """
        if self.global_cm:
            if self.colormap is None:
                raise IOError("if global_cm=True, colormap must be defined")
            else:
                all_colors =\
                    (create_colormap(sf.ravel(), self.colormap) * 255)\
                    .astype(np.uint8)
        elif self.colormap is not None:
            all_colors =\
                (np.array([create_colormap(sf_i, self.colormap)
                           for sf_i in sf]) * 255).reshape(-1, 3)\
                .astype(np.uint8)
        else:
            all_colors =\
                np.tile(np.abs(self.vertices)*255, (len(sf), 1))\
                .astype(np.uint8)
        return all_colors
